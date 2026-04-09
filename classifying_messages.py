import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
import ast

logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

litellm_api_key = os.getenv("LITELLM_API_KEY")
litellm_api_base = os.getenv("LITELLM_API_BASE")
#client = OpenAI(api_key=litellm_api_key, base_url=litellm_api_base)

NO_USER_MESSAGE_AVAILABLE = "No user message available"

RECLASSIFIED_CATEGORY_DESCRIPTION = "Reclassified Category Description"

categories = {
    1: "Inhaltliche Fragen zum Vorlesungsstoff (das sind Fragen 'was sind delkredere?' 'löse mir diese aufgabe', etc.)",
    2: "Administrative Fragen zu einzelnen Vorlesungen / Seminaren (z.B. 'wann findet die Prüfung statt?', 'Welche Themen kommen in der Vorlesung vor?', 'Was ist, wenn ich eine Prüfung nicht bestehe?')",
    3: "Administrative Fragen zum Studium (Pflichtfächer, Kreditsanrechnen, Austauschsemester, 'welche Fächer muss ich nehmen?', 'Kann ich von BWL auf VWL wechseln?')",
    4: "Administrative Fragen zur Bachelorarbeit (Deadline, Organisation, Prozess)",
    5: "Stundenplan oder Lehrplan (erstellen, anpassen, etc.)",
    6: "Fragen zur UZH (zu Studierendenvereinen, Gebäuden, Freizeitangeboten, etc.)",
    7: "Fragen zu IT (z.B. 'Wo finde ich mein Passwort?', 'Wie kann ich drucken?')",
    8: "Fragen zu Software (Excel, Klicker, OLAT, etc.)",
    9: "Fragen zu Health (z.B. 'mir geht’s nicht gut', 'ich fühle, dass es zu viel wird', etc.)",
    10: "Fragen zu Career / Berufseinstieg / Praktikum (z.B. 'wie bereite ich mich auf ein Interview vor?')",
    11: "Begrüssung (z.B. 'hallo', etc.)",
    12: "Interaktionsbefehl (z.B. 'antworte', 'nochmals', 'was ist das?')"
}

categories_extended = categories.copy()
categories_extended[-1] = NO_USER_MESSAGE_AVAILABLE
categories_extended[0] = "No category matched"

log = logging.getLogger(__name__)

def classify_messages_by_LLM(df, excel_file_name: str, pickle_file_name: str, batch_size: int, messageColumnName: str = "User Question"):
    """
    Incrementally classify messages using an LLM and update the Excel file.

    :param df: Pandas DataFrame containing conversations and messages.
    :param excel_file_name: Path to the Excel file to overwrite.
    :param batch_size: Number of unclassified messages to process in this batch.
    :param messageColumnName: Name of the column in df that contains the messages to classify.
    """
    log.info("Starting incremental LLM classification...")

    extracted_message_col = "Extracted Message"
    reclassified_category_id_col = "Reclassified Category Id"

    # Add missing columns if they don’t exist yet
    if extracted_message_col not in df.columns:
        df[extracted_message_col] = None
    if reclassified_category_id_col not in df.columns:
        df[reclassified_category_id_col] = None
    if RECLASSIFIED_CATEGORY_DESCRIPTION not in df.columns:
        df[RECLASSIFIED_CATEGORY_DESCRIPTION] = None

    # Select unclassified rows
    unclassified_mask = df[reclassified_category_id_col].isna()
    unclassified_df = df[unclassified_mask]

    if unclassified_df.empty:
        log.info("All messages are already classified.")
        return

    # Limit to batch size
    if batch_size is not None:
        batch_df = unclassified_df.head(batch_size).copy()
        log.info(f"Classifying {len(batch_df)} messages out of {len(unclassified_df)} unclassified...")
    else:
        batch_df = unclassified_df.copy()
        log.info(f"Classifying all {len(batch_df)} messages ...")

    # Perform classification
    batch_df[extracted_message_col] = batch_df[messageColumnName].apply(lambda x: extract_message(x, messageColumnName))
    batch_df[reclassified_category_id_col] = batch_df[extracted_message_col].apply(lambda x: classify_message(x, categories))
    batch_df[RECLASSIFIED_CATEGORY_DESCRIPTION] = batch_df[reclassified_category_id_col].map(categories_extended).fillna("No category id matched!")

    # Update original dataframe
    df.update(batch_df)

    # Save back to Excel
    if excel_file_name is not None:
        df.to_excel(excel_file_name, index=False)
        log.info(f"Updated Excel file saved: {excel_file_name}")
    if pickle_file_name is not None:
        df.to_pickle(pickle_file_name)
        log.info(f"DataFrame saved as Parquet: {pickle_file_name}")

    # Optionally print summary of this batch
    print(batch_df[[extracted_message_col, reclassified_category_id_col, RECLASSIFIED_CATEGORY_DESCRIPTION]].head())

def extract_message(data: str, messageColumnName: str):
    #print(f"\n\nExtracting message from column '{messageColumnName}': {data}...")

    data_dict = ast.literal_eval(data)

    items = data_dict if isinstance(data_dict, list) else [data_dict]

    texts = []
    for item in items:
        kwargs = item.get("kwargs", {})

        if kwargs.get("role") == "user":
            contents = kwargs.get("content", [])

            for c in contents:
                if c.get("type") == "text":
                    texts.append(c.get("text"))

    #print(f"Extracted message(s): {texts}...")
    return " ".join(texts) if texts else None

def classify_message(message: str, categories: dict) -> str:
    """
    Classify a single message using an LLM.
    :return: classification result (category ID) or 0 when no category could be determined or -1 when no message available
    """
    if message is None:
        print("\n\nNo message to classify.")
        return -1

    print(f"\n\nTry to classify message: {message}")

    categories_str = "\n".join([f"{cid}: {desc}" for cid, desc in categories.items()])

    prompt = f"""
Classify the following user question into one of these numbered categories.
Return only the category number (integer). If no category matches, return 0.

Categories:
{categories_str}

User question: "{message}"
"""
    #log.info(f"PROMPT: {prompt}")

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    classification = response.choices[0].message.content.strip()

    # Try to convert to int, fallback to 0
    try:
        category_id = int(classification)
    except ValueError:
        category_id = 0 # Could not be categorized

    print(f"Classification result: {category_id}")

    return category_id

def display_classification_results(df, excluded_user_ids=None, user_id_column_name: str = "User ID"):
    """
    Display classification results
    :param df: Pandas dataframe containing classification results
    :param excluded_user_ids: List of user IDs (strings) to exclude from the analysis
    :param user_id_column_name: Name of column in dataframe containing user IDs, e.g. "User ID" or "userId"
    """
    log.info(f"\n\nDisplay classification results '{RECLASSIFIED_CATEGORY_DESCRIPTION}'...")

    # Exclude specific users if a list is provided
    if excluded_user_ids:
        initial_count = len(df)
        df = df[~df[user_id_column_name].isin(excluded_user_ids)]
        log.info(f"Excluded {len(excluded_user_ids)} user(s) — filtered from {initial_count} to {len(df)} rows.")

    # Exclude all entries where no user message was available
    df = df[df[RECLASSIFIED_CATEGORY_DESCRIPTION] != NO_USER_MESSAGE_AVAILABLE]

    # Check that the column exists and inspect it
    print("\n\n")
    print(df[RECLASSIFIED_CATEGORY_DESCRIPTION].head(12))
    print(df[[user_id_column_name, RECLASSIFIED_CATEGORY_DESCRIPTION]].head(12))

    # Count messages per classification
    category_counts = df[RECLASSIFIED_CATEGORY_DESCRIPTION].value_counts()

    # Convert to DataFrame (optional, for nicer display)
    category_ranking = category_counts.reset_index()
    category_ranking.columns = [RECLASSIFIED_CATEGORY_DESCRIPTION, "Message Count"]
    category_ranking["Percentage"] = category_ranking["Message Count"] / category_ranking["Message Count"].sum() * 100

    # Sort (value_counts already sorts descending, but just to be explicit)
    category_ranking = category_ranking.sort_values(by="Message Count", ascending=False)
    print("\n\n")
    print(category_ranking)