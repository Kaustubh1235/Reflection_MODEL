import pandas as pd
import re
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# 1. Path to your local JSON file.
#    '../' means go up one directory from the 'src' folder.
INPUT_JSON_PATH = "constitution_qa.json"

# 2. The exact name of the column/key in your JSON that contains the QUESTION.
QUESTION_COLUMN = "question"

# 3. The exact name of the column/key in your JSON that contains the ANSWER.
ANSWER_COLUMN = "answer"

# 4. The name of the output file.
OUTPUT_FILENAME = "formatted_qa_data.csv"

# 5. The number of words from the question to use for the generated tag.
TAG_WORD_COUNT = 3

# ==============================================================================

def generate_generic_tag(question_text, word_count=3):
    """
    Generates a generic, clean tag from the first few words of a question.
    """
    if not isinstance(question_text, str):
        return "[ unknown-format ]"
        
    words = re.findall(r'\b\w+\b', question_text.lower())
    tag_text = "-".join(words[:word_count])
    return f"[ {tag_text} ]"

def format_local_json_to_csv():
    """
    Loads a local JSON QA dataset, formats it, and saves it as a CSV.
    """
    print(f"🚀 Starting conversion for local file: {INPUT_JSON_PATH}")
    
    try:
        # 1. Check if the file exists
        if not os.path.exists(INPUT_JSON_PATH):
            raise FileNotFoundError(f"Error: The file '{INPUT_JSON_PATH}' was not found. Make sure it's in your main project folder.")

        # 2. Load the local JSON file into a pandas DataFrame
        df = pd.read_json(INPUT_JSON_PATH)
        
        if df is None or df.empty:
            print("❌ DataFrame is empty. The JSON file might be malformed. Exiting.")
            return

        print(f"✅ File loaded successfully with {len(df)} records.")
        print(f"Columns found: {list(df.columns)}")

        # 3. Validate that the specified columns exist
        if QUESTION_COLUMN not in df.columns:
            raise ValueError(f"Error: Question column '{QUESTION_COLUMN}' not found in the JSON!")
        if ANSWER_COLUMN not in df.columns:
            raise ValueError(f"Error: Answer column '{ANSWER_COLUMN}' not found in the JSON!")

        print(f"Using '{QUESTION_COLUMN}' for questions and '{ANSWER_COLUMN}' for answers.")

        # 4. Create the new formatted column
        print("Generating tags and formatting the data...")
        df.dropna(subset=[QUESTION_COLUMN, ANSWER_COLUMN], inplace=True)
        df['formatted_question'] = df[QUESTION_COLUMN].apply(
            lambda q: f"{generate_generic_tag(q, TAG_WORD_COUNT)} {q}"
        )

        # 5. Create the final DataFrame
        final_df = df[['formatted_question', ANSWER_COLUMN]]

        # 6. Save the final result to a CSV file in the main directory
        output_path = os.path.join("..", OUTPUT_FILENAME)
        print(f"💾 Saving the formatted data to '{output_path}'...")
        
        final_df.to_csv(
            output_path,
            header=False,
            index=False,
            sep=',',
            quoting=1 
        )

        print(f"\n🎉 Conversion complete! File '{OUTPUT_FILENAME}' has been created in your main project folder.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    format_local_json_to_csv()