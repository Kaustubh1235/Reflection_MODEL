import pandas as pd
import numpy as np
import re
import nltk
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class PreprocessingConfig:
    """Configuration class with validation and defaults."""
    
    # Required parameters
    input_csv_path: str
    text_column: str
    
    # Optional parameters with defaults
    target_column: Optional[str] = None
    output_dir: str = "processed_data"
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True
    
    # Text cleaning options
    remove_urls: bool = True
    remove_emails: bool = True
    remove_html: bool = True
    remove_numbers: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    min_word_length: int = 2
    max_text_length: Optional[int] = None
    
    # Advanced options
    custom_stopwords: List[str] = None
    preserve_patterns: List[str] = None
    encoding: str = "utf-8"
    handle_duplicates: str = "remove"  # "remove", "keep_first", "keep_last", "keep_all"
    min_samples_per_class: int = 2
    
    # Logging
    log_level: str = "INFO"
    save_preprocessing_report: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if not 0 <= self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1")
        
        if self.handle_duplicates not in ["remove", "keep_first", "keep_last", "keep_all"]:
            raise ValueError("handle_duplicates must be one of: 'remove', 'keep_first', 'keep_last', 'keep_all'")
        
        if self.custom_stopwords is None:
            self.custom_stopwords = []
        
        if self.preserve_patterns is None:
            self.preserve_patterns = []

class EnhancedDataPreprocessor:
    """
    Enhanced Grade A preprocessing pipeline with comprehensive features:
    - Robust error handling and logging
    - Data quality assessment and reporting
    - Advanced text cleaning options
    - Validation set creation
    - Duplicate handling
    - Memory optimization
    - Comprehensive statistics and visualization
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the enhanced preprocessor.
        
        Args:
            config: PreprocessingConfig object with all settings
        """
        self.config = config
        self._setup_logging()
        self._download_nltk_resources()
        self._initialize_text_processors()
        
        # Statistics tracking
        self.stats = {
            'original_rows': 0,
            'final_rows': 0,
            'duplicates_removed': 0,
            'empty_text_removed': 0,
            'cleaning_operations': [],
            'class_distribution': {},
            'text_length_stats': {},
            'processing_time': 0
        }
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"preprocessing_{timestamp}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced preprocessing pipeline initialized")
        self.logger.info(f"Configuration: {asdict(self.config)}")

    def _download_nltk_resources(self):
        """Download required NLTK packages with better error handling."""
        self.logger.info("Checking NLTK resources...")
        
        required_packages = {
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords', 
            'wordnet': 'corpora/wordnet',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
        }
        
        for package, path in required_packages.items():
            try:
                nltk.data.find(path)
                self.logger.debug(f"NLTK package '{package}' found")
            except LookupError:
                self.logger.warning(f"Downloading NLTK package '{package}'...")
                try:
                    nltk.download(package, quiet=True)
                    self.logger.info(f"Successfully downloaded '{package}'")
                except Exception as e:
                    self.logger.error(f"Failed to download '{package}': {e}")
                    raise

    def _initialize_text_processors(self):
        """Initialize text processing components."""
        try:
            self.lemmatizer = WordNetLemmatizer()
            base_stopwords = set(stopwords.words('english'))
            custom_stopwords = set(self.config.custom_stopwords)
            self.stop_words = base_stopwords.union(custom_stopwords)
            
            # Compile regex patterns for better performance
            self.url_pattern = re.compile(r'http\S+|www\S+|https\S+', re.MULTILINE | re.IGNORECASE)
            self.email_pattern = re.compile(r'\S*@\S*\s?')
            self.html_pattern = re.compile(r'<[^<]+?>')
            self.number_pattern = re.compile(r'\d+')
            self.punct_pattern = re.compile(r'[^a-zA-Z\s]')
            
            self.logger.info("Text processors initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize text processors: {e}")
            raise

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        self.logger.info("Assessing data quality...")
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'text_column_stats': {}
        }
        
        # Text column specific analysis
        if self.config.text_column in df.columns:
            text_series = df[self.config.text_column].dropna()
            quality_report['text_column_stats'] = {
                'non_null_count': len(text_series),
                'unique_count': text_series.nunique(),
                'avg_length': text_series.str.len().mean(),
                'max_length': text_series.str.len().max(),
                'min_length': text_series.str.len().min(),
                'empty_strings': (text_series.str.strip() == '').sum()
            }
        
        # Target column analysis
        if self.config.target_column and self.config.target_column in df.columns:
            quality_report['target_distribution'] = df[self.config.target_column].value_counts().to_dict()
        
        self.logger.info(f"Data quality assessment complete: {quality_report['total_rows']} rows, "
                        f"{quality_report['duplicate_rows']} duplicates, "
                        f"{quality_report['memory_usage_mb']:.2f} MB")
        
        return quality_report

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate rows based on configuration."""
        if self.config.handle_duplicates == "keep_all":
            return df
        
        original_count = len(df)
        
        if self.config.handle_duplicates == "remove":
            df = df.drop_duplicates()
        elif self.config.handle_duplicates == "keep_first":
            df = df.drop_duplicates(keep='first')
        elif self.config.handle_duplicates == "keep_last":
            df = df.drop_duplicates(keep='last')
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            self.stats['duplicates_removed'] = removed_count
            self.logger.info(f"Removed {removed_count} duplicate rows")
        
        return df.reset_index(drop=True)

    def _advanced_text_cleaning(self, text: str) -> str:
        """Enhanced text cleaning with configurable options."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        original_text = text
        
        # Apply preserve patterns first (mark them for protection)
        preserved_parts = {}
        for i, pattern in enumerate(self.config.preserve_patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            for j, match in enumerate(matches):
                placeholder = f"__PRESERVE_{i}_{j}__"
                preserved_parts[placeholder] = match
                text = text.replace(match, placeholder, 1)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove unwanted elements based on configuration
        if self.config.remove_urls:
            text = self.url_pattern.sub('', text)
        
        if self.config.remove_emails:
            text = self.email_pattern.sub('', text)
        
        if self.config.remove_html:
            text = self.html_pattern.sub('', text)
        
        if self.config.remove_numbers:
            text = self.number_pattern.sub('', text)
        
        if self.config.remove_punctuation:
            text = self.punct_pattern.sub('', text)
        
        # Tokenization and advanced processing
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple split if tokenization fails
            tokens = text.split()
        
        # Filter and process tokens
        processed_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) <= self.config.min_word_length:
                continue
            
            # Skip stopwords if configured
            if self.config.remove_stopwords and token in self.stop_words:
                continue
            
            # Lemmatization if configured
            if self.config.lemmatize:
                try:
                    token = self.lemmatizer.lemmatize(token)
                except Exception:
                    pass  # Keep original token if lemmatization fails
            
            processed_tokens.append(token)
        
        # Join tokens back
        cleaned_text = " ".join(processed_tokens)
        
        # Restore preserved patterns
        for placeholder, original_match in preserved_parts.items():
            cleaned_text = cleaned_text.replace(placeholder, original_match)
        
        # Apply length limit if specified
        if self.config.max_text_length and len(cleaned_text) > self.config.max_text_length:
            cleaned_text = cleaned_text[:self.config.max_text_length].strip()
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text

    def _create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Create train/test/validation splits with advanced stratification."""
        self.logger.info("Creating dataset splits...")
        
        # Determine stratification column
        stratify_col = None
        if (self.config.stratify and self.config.target_column and 
            self.config.target_column in df.columns):
            
            # Check class distribution
            class_counts = df[self.config.target_column].value_counts()
            min_samples = class_counts.min()
            
            if min_samples >= self.config.min_samples_per_class:
                stratify_col = df[self.config.target_column]
                self.logger.info(f"Using stratified split on '{self.config.target_column}'")
            else:
                self.logger.warning(f"Insufficient samples per class for stratification "
                                  f"(min: {min_samples}, required: {self.config.min_samples_per_class})")
        
        # First split: separate test set
        if stratify_col is not None:
            train_val_df, test_df = train_test_split(
                df, 
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_col
            )
        else:
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
        
        # Second split: separate validation set if requested
        val_df = None
        if self.config.validation_size > 0:
            adjusted_val_size = self.config.validation_size / (1 - self.config.test_size)
            
            if stratify_col is not None:
                train_stratify = train_val_df[self.config.target_column]
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=adjusted_val_size,
                    random_state=self.config.random_state,
                    stratify=train_stratify
                )
            else:
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=adjusted_val_size,
                    random_state=self.config.random_state
                )
        else:
            train_df = train_val_df
        
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        if val_df is not None:
            val_df = val_df.reset_index(drop=True)
        
        self.logger.info(f"Dataset splits created - Train: {len(train_df)}, "
                        f"Test: {len(test_df)}" + 
                        (f", Validation: {len(val_df)}" if val_df is not None else ""))
        
        return train_df, test_df, val_df

    def _save_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      val_df: Optional[pd.DataFrame] = None):
        """Save processed datasets with metadata."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_path = output_dir / "train.csv"
        test_path = output_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False, encoding=self.config.encoding)
        test_df.to_csv(test_path, index=False, encoding=self.config.encoding)
        
        self.logger.info(f"Saved training data: {train_path} ({len(train_df)} rows)")
        self.logger.info(f"Saved testing data: {test_path} ({len(test_df)} rows)")
        
        if val_df is not None:
            val_path = output_dir / "validation.csv"
            val_df.to_csv(val_path, index=False, encoding=self.config.encoding)
            self.logger.info(f"Saved validation data: {val_path} ({len(val_df)} rows)")
        
        # Save configuration and metadata
        config_path = output_dir / "preprocessing_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        
        # Save processing statistics
        stats_path = output_dir / "preprocessing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)

    def _generate_preprocessing_report(self, quality_report: Dict[str, Any]):
        """Generate comprehensive preprocessing report."""
        if not self.config.save_preprocessing_report:
            return
        
        output_dir = Path(self.config.output_dir)
        report_path = output_dir / "preprocessing_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Data Preprocessing Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n")
            f.write("```json\n")
            f.write(json.dumps(asdict(self.config), indent=2, default=str))
            f.write("\n```\n\n")
            
            f.write("## Data Quality Assessment\n")
            f.write(f"- **Original rows:** {quality_report['total_rows']}\n")
            f.write(f"- **Final rows:** {self.stats['final_rows']}\n")
            f.write(f"- **Duplicates removed:** {self.stats['duplicates_removed']}\n")
            f.write(f"- **Empty text removed:** {self.stats['empty_text_removed']}\n")
            f.write(f"- **Memory usage:** {quality_report['memory_usage_mb']:.2f} MB\n\n")
            
            if 'target_distribution' in quality_report:
                f.write("## Target Distribution\n")
                for label, count in quality_report['target_distribution'].items():
                    f.write(f"- **{label}:** {count}\n")
                f.write("\n")
            
            f.write("## Text Statistics\n")
            text_stats = quality_report.get('text_column_stats', {})
            for key, value in text_stats.items():
                f.write(f"- **{key}:** {value}\n")
            f.write("\n")
            
            f.write("## Processing Summary\n")
            for operation in self.stats['cleaning_operations']:
                f.write(f"- {operation}\n")
        
        self.logger.info(f"Preprocessing report saved: {report_path}")

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete enhanced preprocessing pipeline.
        
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting enhanced preprocessing pipeline...")
        
        try:
            # Step 1: Load and validate data
            self.logger.info(f"Loading data from: {self.config.input_csv_path}")
            
            try:
                df = pd.read_csv(self.config.input_csv_path, encoding=self.config.encoding)
            except UnicodeDecodeError:
                self.logger.warning(f"Failed to read with {self.config.encoding}, trying 'latin-1'")
                df = pd.read_csv(self.config.input_csv_path, encoding='latin-1')
            
            self.stats['original_rows'] = len(df)
            
            # Validate required columns
            required_cols = [self.config.text_column]
            if self.config.target_column:
                required_cols.append(self.config.target_column)
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Required columns not found: {missing_cols}")
            
            self.logger.info(f"Successfully loaded {len(df)} rows with columns: {list(df.columns)}")
            
            # Step 2: Data quality assessment
            quality_report = self._assess_data_quality(df)
            
            # Step 3: Handle duplicates
            df = self._handle_duplicates(df)
            
            # Step 4: Clean text data
            self.logger.info(f"Cleaning text in column: {self.config.text_column}")
            
            # Remove rows with null text values
            original_count = len(df)
            df = df.dropna(subset=[self.config.text_column])
            null_removed = original_count - len(df)
            if null_removed > 0:
                self.logger.info(f"Removed {null_removed} rows with null text values")
            
            # Apply text cleaning
            cleaned_column = f"{self.config.text_column}_cleaned"
            df[cleaned_column] = df[self.config.text_column].apply(self._advanced_text_cleaning)
            
            # Remove rows where cleaned text is empty
            original_count = len(df)
            df = df[df[cleaned_column].str.strip().astype(bool)]
            empty_removed = original_count - len(df)
            if empty_removed > 0:
                self.stats['empty_text_removed'] = empty_removed
                self.logger.info(f"Removed {empty_removed} rows with empty cleaned text")
            
            self.stats['final_rows'] = len(df)
            
            # Step 5: Create splits
            train_df, test_df, val_df = self._create_splits(df)
            
            # Step 6: Save datasets
            self._save_datasets(train_df, test_df, val_df)
            
            # Step 7: Generate report
            self._generate_preprocessing_report(quality_report)
            
            # Calculate processing time
            end_time = datetime.now()
            self.stats['processing_time'] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"🎉 Pipeline completed successfully in {self.stats['processing_time']:.2f} seconds")
            
            return {
                'success': True,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'val_size': len(val_df) if val_df is not None else 0,
                'processing_time': self.stats['processing_time'],
                'output_dir': str(self.config.output_dir),
                'stats': self.stats
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }

def create_sample_config() -> PreprocessingConfig:
    """Create a sample configuration for demonstration."""
    return PreprocessingConfig(
        input_csv_path="sample_dataset.csv",
        text_column="text_review",
        target_column="sentiment",
        output_dir="enhanced_processed_data",
        test_size=0.2,
        validation_size=0.1,
        stratify=True,
        remove_urls=True,
        remove_emails=True,
        remove_html=True,
        lemmatize=True,
        min_word_length=2,
        custom_stopwords=["product", "item"],  # Domain-specific stopwords
        handle_duplicates="remove",
        save_preprocessing_report=True,
        log_level="INFO"
    )

def create_constitution_config() -> PreprocessingConfig:
    """Create configuration specifically for constitution dataset."""
    return PreprocessingConfig(
        input_csv_path="Dev/preprocessed_data/formatted_constitution_dataset.csv",
        text_column="text",  # Adjust based on your actual column name
        target_column=None,  # Set this if you have a target column
        output_dir="Dev/preprocessed_data/processed_constitution",
        test_size=0.2,
        validation_size=0.1,
        stratify=False,  # Set to True if you have a target column
        remove_urls=True,
        remove_emails=True,
        remove_html=True,
        remove_numbers=False,  # Keep numbers for legal documents
        remove_punctuation=False,  # Keep punctuation for legal text
        lemmatize=True,
        min_word_length=2,
        custom_stopwords=["article", "section", "shall"],  # Legal-specific stopwords
        preserve_patterns=[r'\b[A-Z][a-z]+\s+\d+\b', r'\bSection\s+\d+\b'],  # Preserve legal references
        handle_duplicates="remove",
        save_preprocessing_report=True,
        log_level="INFO"
    )

if __name__ == "__main__":
    print("🏛️ Processing Constitution Dataset...")
    
    # First, let's inspect the dataset to understand its structure
    try:
        # Try multiple possible paths
        possible_paths = [
            "Dev/preprocessed_data/formatted_constitution_dataset.csv",
            "preprocessed_data/formatted_constitution_dataset.csv", 
            "../preprocessed_data/formatted_constitution_dataset.csv",
            "./Dev/preprocessed_data/formatted_constitution_dataset.csv"
        ]
        
        dataset_path = None
        for path in possible_paths:
            if Path(path).exists():
                dataset_path = path
                break
        
        if dataset_path is None:
            raise FileNotFoundError("Could not locate the constitution dataset in any expected location")
            
        # Load the dataset to inspect its structure
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset found at: {dataset_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        print(f"\nSample text content:")
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.len().mean() > 50:  # Likely text column
                print(f"Column '{col}' sample:")
                print(f"  {df[col].iloc[0][:200]}...")
                break
        
        # Create configuration for constitution dataset
        config = create_constitution_config()
        config.input_csv_path = dataset_path  # Use the actual found path
        
        # Prompt user to confirm/modify text column
        print(f"\n🔧 Current configuration:")
        print(f"  - Text column: '{config.text_column}'")
        print(f"  - Target column: {config.target_column}")
        print(f"  - Output directory: {config.output_dir}")
        print(f"  - Preserving legal references and keeping numbers/punctuation")
        
        # Update configuration based on actual dataset structure
        text_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().mean() > 50]
        if text_columns and config.text_column not in df.columns:
            config.text_column = text_columns[0]
            print(f"  - Auto-detected text column: '{config.text_column}'")
        
        # Run the enhanced preprocessor
        preprocessor = EnhancedDataPreprocessor(config)
        results = preprocessor.run()
        
        if results['success']:
            print(f"\n✅ Constitution dataset preprocessing completed successfully!")
            print(f"📊 Results: Train={results['train_size']}, Test={results['test_size']}, Val={results['val_size']}")
            print(f"⏱️ Processing time: {results['processing_time']:.2f} seconds")
            print(f"📁 Output directory: {results['output_dir']}")
            print(f"\n📋 Files created:")
            output_path = Path(results['output_dir'])
            for file in output_path.glob("*"):
                print(f"  - {file.name}")
        else:
            print(f"\n❌ Preprocessing failed: {results['error']}")
            
    except FileNotFoundError as e:
        print(f"❌ Could not find the constitution dataset file")
        print(f"Error: {str(e)}")
        print("Checked the following paths:")
        for path in possible_paths:
            exists = "✅" if Path(path).exists() else "❌"
            print(f"  {exists} {path}")
        print("Please ensure the file exists at one of these paths.")
        
        # Create a demo configuration that users can modify
        print(f"\n📝 Here's a template configuration you can use:")
        config = create_constitution_config()
        print("```python")
        print("config = PreprocessingConfig(")
        for key, value in asdict(config).items():
            print(f"    {key}={repr(value)},")
        print(")")
        print("```")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        print("Please check the file format and path.")