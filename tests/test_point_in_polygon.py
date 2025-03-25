"""Tests for Point in Polygon functionality."""

import os
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from point_in_polygon import PointInPolygonConfig, point_in_polygon

import os
# Set Oracle environment variables at the start of your script
os.environ['ORACLE_HOME'] = r"C:\ora19c\product\19.0.0\client_2"
os.environ['PATH'] = os.environ.get('PATH', '') + r";C:\ora19c\product\19.0.0\client_2\bin"
os.environ['TNS_ADMIN'] = r"C:\or"

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file with test coordinates."""
    csv_path = tmp_path / "test_points.csv"
    df = pd.DataFrame({
        'rec_id': [1, 2, 3],
        'latitude': [45.4215, 45.4216, 45.4217],
        'longitude': [-75.6972, -75.6973, -75.6974]
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def output_csv(tmp_path):
    """Create an output CSV path."""
    return str(tmp_path / "output.csv")


@pytest.fixture
def cache_dir(tmp_path):
    """Create a cache directory."""
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    return str(cache_path)


class TestPointInPolygonConfig:
    """Test cases for PointInPolygonConfig."""

    def test_valid_config(self, sample_csv, output_csv, cache_dir):
        """Test creation of config with valid parameters."""
        config = PointInPolygonConfig(
            csv_long_lat_file=sample_csv,
            output_csv_file=output_csv,
            table_name="test_table",
            cache_dir=cache_dir
        )
        assert config.hostname == "Geodepot"
        assert config.database_name == "WAREHOUSE"
        assert config.table_name == "test_table"
        assert config.use_parallel is True

    def test_missing_required_fields(self):
        """Test that config fails when required fields are missing."""
        with pytest.raises(ValidationError):
            PointInPolygonConfig()

    def test_invalid_csv_file(self, output_csv):
        """Test that config fails with non-existent CSV file."""
        with pytest.raises(ValidationError):
            PointInPolygonConfig(
                csv_long_lat_file="nonexistent.csv",
                output_csv_file=output_csv,
                table_name="test_table"
            )

    def test_invalid_column_names(self, sample_csv, output_csv):
        """Test validation of CSV column names."""
        with pytest.raises(ValidationError):
            config = PointInPolygonConfig(
                csv_long_lat_file=sample_csv,
                output_csv_file=output_csv,
                table_name="test_table",
                id_column="invalid_id",
                lat_column="invalid_lat",
                lon_column="invalid_lon"
            )
            # Trigger validation by accessing the file
            with open(config.csv_long_lat_file) as f:
                pass


@pytest.fixture
def mock_gdf():
    """Create a mock GeoDataFrame for testing."""
    return MagicMock()


@pytest.fixture
def basic_config(sample_csv, output_csv, cache_dir):
    """Create a basic config for testing."""
    return PointInPolygonConfig(
        csv_long_lat_file=sample_csv,
        output_csv_file=output_csv,
        table_name="test_table",
        cache_dir=cache_dir
    )


class TestPointInPolygon:
    """Test cases for point_in_polygon function."""

    @patch('point_in_polygon.main.get_df_shapes')
    @patch('point_in_polygon.main.process_points_parallel')
    def test_successful_execution(self, mock_process, mock_get_shapes, basic_config, mock_gdf):
        """Test successful execution of point_in_polygon."""
        # Setup mocks
        mock_get_shapes.return_value = (mock_gdf, True)
        mock_process.return_value = pd.DataFrame({
            'rec_id': [1, 2, 3],
            'bb_uid_matched': [True, False, True]
        })

        # Run function
        result = point_in_polygon(basic_config)

        # Verify
        assert result is True
        mock_get_shapes.assert_called_once()
        mock_process.assert_called_once()
        assert os.path.exists(basic_config.output_csv_file)

    @patch('point_in_polygon.main.get_df_shapes')
    def test_database_connection_error(self, mock_get_shapes, basic_config):
        """Test handling of database connection errors."""
        mock_get_shapes.side_effect = Exception("Database connection failed")

        with pytest.raises(Exception) as exc_info:
            point_in_polygon(basic_config)
        assert "Database connection failed" in str(exc_info.value)

    def test_caching_behavior(self, basic_config):
        """Test that caching works as expected."""
        # First run should create cache
        with patch('point_in_polygon.main.get_df_shapes') as mock_get_shapes:
            mock_get_shapes.return_value = (MagicMock(), True)
            point_in_polygon(basic_config)

        # Verify cache directory contains files
        cache_files = list(Path(basic_config.cache_dir).glob("*"))
        assert len(cache_files) > 0

    @patch('point_in_polygon.main.get_df_shapes')
    @patch('point_in_polygon.main.process_points_parallel')
    def test_return_all_points_true(self, mock_process, mock_get_shapes, basic_config, mock_gdf):
        """Test that all points are returned when return_all_points is True."""
        mock_get_shapes.return_value = (mock_gdf, True)
        df = pd.DataFrame({
            'rec_id': [1, 2, 3],
            'bb_uid_matched': [True, False, True]
        })
        mock_process.return_value = df

        basic_config.return_all_points = True
        point_in_polygon(basic_config)

        # Read output file and verify all points are present
        output_df = pd.read_csv(basic_config.output_csv_file)
        assert len(output_df) == 3

    @patch('point_in_polygon.main.get_df_shapes')
    @patch('point_in_polygon.main.process_points_parallel')
    def test_return_all_points_false(self, mock_process, mock_get_shapes, basic_config, mock_gdf):
        """Test that only matched points are returned when return_all_points is False."""
        mock_get_shapes.return_value = (mock_gdf, True)
        df = pd.DataFrame({
            'rec_id': [1, 2, 3],
            'bb_uid_matched': [True, False, True]
        })
        mock_process.return_value = df

        basic_config.return_all_points = False
        point_in_polygon(basic_config)

        # Read output file and verify only matched points are present
        output_df = pd.read_csv(basic_config.output_csv_file)
        assert len(output_df) == 2  # Only the True matches

    def test_parallel_processing(self, basic_config):
        """Test that parallel processing is used when configured."""
        basic_config.use_parallel = True
        basic_config.max_workers = 2

        with patch('point_in_polygon.main.get_df_shapes') as mock_get_shapes, \
             patch('point_in_polygon.main.process_points_parallel') as mock_process:
            mock_get_shapes.return_value = (MagicMock(), True)
            mock_process.return_value = pd.DataFrame()
            
            point_in_polygon(basic_config)
            
            # Verify parallel processing was used
            mock_process.assert_called_once()

    def test_chunk_processing(self, basic_config):
        """Test that data is processed in chunks."""
        basic_config.chunk_size = 2  # Small chunk size for testing

        with patch('point_in_polygon.main.get_df_shapes') as mock_get_shapes, \
             patch('point_in_polygon.main.process_points_parallel') as mock_process:
            mock_get_shapes.return_value = (MagicMock(), True)
            mock_process.return_value = pd.DataFrame()
            
            point_in_polygon(basic_config)
            
            # Verify chunked processing
            calls = mock_process.call_args_list
            for call in calls:
                chunk = call[0][0]  # First argument of each call
                assert len(chunk) <= basic_config.chunk_size


def test_integration(tmp_path):
    """Integration test using a small dataset."""
    # Create test data
    input_csv = tmp_path / "test_points.csv"
    pd.DataFrame({
        'rec_id': [1, 2],
        'latitude': [45.4215, 45.4216],
        'longitude': [-75.6972, -75.6973]
    }).to_csv(input_csv, index=False)

    output_csv = tmp_path / "output.csv"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    config = PointInPolygonConfig(
        csv_long_lat_file=str(input_csv),
        output_csv_file=str(output_csv),
        table_name="test_table",
        cache_dir=str(cache_dir),
        sample=True  # Use sample mode for testing
    )

    with patch('point_in_polygon.main.get_df_shapes') as mock_get_shapes:
        # Mock the database response
        mock_gdf = MagicMock()
        mock_get_shapes.return_value = (mock_gdf, True)

        # Run the function
        result = point_in_polygon(config)

        # Verify results
        assert result is True
        assert output_csv.exists()
        output_df = pd.read_csv(output_csv)
        assert len(output_df) > 0
        assert 'bb_uid_matched' in output_df.columns
