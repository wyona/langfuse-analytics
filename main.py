import glob
import logging
import pandas as pd
import os
from langfuse import Langfuse
from dotenv import load_dotenv
#import matplotlib.pyplot as plt
from ranking_by_conversations_and_messages_by_user import ranking_by_conversations_and_messages_by_user
from conversations_and_messages_per_day import conversations_and_messages_per_day, conversations_and_messages_of_user_per_day
from classifying_messages import classify_messages_by_LLM, display_classification_results

logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

messages_log_file = "Classified_Prod_Messages_Export_16_12_2025.xlsx"
#messages_log_file = "Classified_Prod_Messages_Export_05_11_2025.xlsx"
#messages_log_file = "Classified_Prod_Messages_Export_20_10_2025.xlsx"

pickle_file_name = "Classified_Prod_Messages_Export_07_04_2026_12_00.pkl"

LF_COLUMN_NAME_USER_ID = "userId"
LF_COLUMN_NAME_DATE = "timestamp"
LF_COLUMN_NAME_MESSAGE = "input.messages"
LF_COLUMN_NAME_CONVERSATION = "sessionId"

TRACES_FILE_GLOB = "langfuse_traces_*.csv"

load_dotenv()
lf_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "Replace_by_your_Langfuse_public_key_when_not_configured_in_env_file")
lf_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "Replace_by_your_Langfuse_secret_key_when_not_configured_in_env_file")
lf_host = os.getenv("LANGFUSE_HOST", "Replace_by_your_Langfuse_host_when_not_configured_in_env_file")

def extract_some_metadata(df: pd.DataFrame, userIdColumnName: str, conversationIdColumnName: str, dateColumnName: str):
    """
    Extract some metadata, e.g., number of conversations and number of users
    :param df: Pandas DataFrame containing conversations and messages
    :param userIdColumnName: Column name of user ID, e.g. "User ID" or "userId"
    :param conversationIdColumnName: Column name of conversation ID, e.g. "Conversation ID" or "sessionId"
    :param dateColumnName: Column name of date, e.g. "Date"
    :return: Number of unique users
    """
    print(df.columns)
    if df.empty:
        log.warning(f"DataFrame is empty")
        return 0

    if dateColumnName in df.columns:
        # TODO: Check why .loc does (not) work
        #df.loc[:, dateColumnName] = pd.to_datetime(df[dateColumnName], format="mixed")
        df[dateColumnName] = pd.to_datetime(df[dateColumnName], format="mixed", utc=True)
        oldest_date = df[dateColumnName].min()
        youngest_date = df[dateColumnName].max()
        log.debug(f"Oldest date: {oldest_date}")
        log.debug(f"Youngest date: {youngest_date}")
        log.info(f"\n\nExtracting some metadata for the time period {oldest_date} to {youngest_date} ...\n")
    else:
        log.warn(f"No such date column: {dateColumnName}")

    row_count = len(df)
    log.info(f"Number of rows: {row_count}")

    unique_conversations = df[conversationIdColumnName].nunique()
    log.info(f"Number of unique conversations: {unique_conversations}")

    unique_users = df[userIdColumnName].nunique()
    log.info(f"Number of unique users: {unique_users}")

    return unique_users

def find_latest_traces_file():
    """
    Find the most recent langfuse traces dump file.
    Files are named: langfuse_traces_YYYY-MM-DD_to_YYYY-MM-DD.csv
    Returns (file_path, newest_timestamp) or (None, None) if no file is found.
    """
    files = glob.glob(TRACES_FILE_GLOB)
    if not files:
        return None, None

    def get_to_date(f):
        base = os.path.basename(f).replace(".csv", "")
        parts = base.split("_to_")
        return parts[-1] if len(parts) >= 2 else ""

    files.sort(key=get_to_date, reverse=True)
    latest_file = files[0]

    df = pd.read_csv(latest_file, usecols=[LF_COLUMN_NAME_DATE])
    df[LF_COLUMN_NAME_DATE] = pd.to_datetime(df[LF_COLUMN_NAME_DATE], format="mixed", utc=True)
    newest_ts = df[LF_COLUMN_NAME_DATE].max()
    log.info(f"Found existing traces file '{latest_file}' (newest trace: {newest_ts})")
    return latest_file, newest_ts

def get_data_from_langfuse(max_pages: int, from_timestamp=None):
    """
    Get data from Langfuse
    :param max_pages: Maximum number of pages to fetch
    :param from_timestamp: If provided, only fetch traces newer than this timestamp (incremental update)
    :return: Dataframe containing conversations and messages
    """
    log.debug(f"LF_public_key: {lf_public_key}")
    log.debug(f"LF_secret_key: {lf_secret_key}")
    log.info(f"LF_host: {lf_host}")
    langfuse = Langfuse(
        public_key=lf_public_key,
        secret_key=lf_secret_key,
        host=lf_host,
        timeout=20,
    )
    # DEBUG Langfuse configuration
    #print(dir(langfuse))
    if from_timestamp is not None:
        log.info(f"Get data from Langfuse: {lf_host} (Max pages: {max_pages}, from: {from_timestamp}) ...")
    else:
        log.info(f"Get data from Langfuse: {lf_host} (Max pages: {max_pages}) ...")

    records = []
    page = 1
    #limit_per_page = 20
    limit_per_page = 100  # 100 traces per page is max allowed by Langfuse

    skipped_traces = 0
    while True:
        if page > max_pages:
            log.info(f"Reached max page limit: {max_pages}")
            break
        else:
            log.info(f"Retrieve page {page} (Max {limit_per_page} traces per page) ...")

        #response = langfuse.api.trace.list(limit=limit_per_page, page=page)
        response = langfuse.api.trace.list(limit=limit_per_page, page=page)
        #response = langfuse.trace.list(limit=limit_per_page, page=page, from_timestamp=from_timestamp)
        # One can filter by tags, but excluding traces with certain tags / negative tagging is not possible
        #response = langfuse_api.trace.list(tags="litellm-internal-health-check", limit=limit_per_page, page=page)
        if not response.data:
            break

        for trace in response.data:
            #print(f"Trace: {trace}")
            user_id = getattr(trace, "user_id", None)
            if not user_id:  # skips None, empty string, etc.
                log.debug(f"Skip {getattr(trace, 'id')} trace with empty user_id (Tags: {trace.tags})")
                skipped_traces += 1
                continue

            if hasattr(trace, "dict"):
                # Pydantic v1
                records.append(trace.dict())
            else:
                # Pydantic v2
                records.append(trace.model_dump())

        page += 1
        if len(response.data) < limit_per_page:  # last page
            log.info(f"No more pages: {len(response.data)} traces retrieved")
            break

    log.info(f"Number of skipped traces (empty '{LF_COLUMN_NAME_USER_ID}'): {skipped_traces}")
    log.info(f"Number of traces with non-empty '{LF_COLUMN_NAME_USER_ID}': {len(records)}")

    if not records:
        log.warning(f"No non-empty traces retrieved!")
        if from_timestamp is not None:
            latest_file, _ = find_latest_traces_file()
            if latest_file:
                log.info(f"Returning existing data from '{latest_file}'.")
                return pd.read_csv(latest_file)
        return pd.DataFrame(columns=["id", LF_COLUMN_NAME_DATE, LF_COLUMN_NAME_USER_ID, LF_COLUMN_NAME_CONVERSATION])

    new_df = pd.json_normalize(records)

    if from_timestamp is not None:
        old_file, _ = find_latest_traces_file()
        if old_file:
            existing_df = pd.read_csv(old_file)
            lf_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["id"])
            log.info(f"Merged {len(new_df)} from Langfuse retrieved traces with {len(existing_df)} existing -> {len(lf_df)} total")
        else:
            lf_df = new_df
            old_file = None
    else:
        lf_df = new_df
        old_file = None

    dates = pd.to_datetime(lf_df[LF_COLUMN_NAME_DATE], format="mixed", utc=True)
    oldest_date = dates.min().date()
    newest_date = dates.max().date()
    filename = f"langfuse_traces_{oldest_date}_to_{newest_date}.csv"
    lf_df.to_csv(filename, index=False)
    log.info(f"Saved {len(lf_df)} traces to '{filename}'")

    if old_file and old_file != filename:
        os.remove(old_file)
        log.info(f"Removed superseded file '{old_file}'")

    print(f"DataFrame Shape: {lf_df.shape}")
    print(lf_df.head())

    return lf_df

def reduce_data_frame(df: pd.DataFrame):
    """
    Reduce data frame to contain conversations and messages without much content
    """
    log.info(f"Available Langfuse traces columns: {df.columns.tolist()}")
    cols = [
        "id",
        LF_COLUMN_NAME_DATE,
        LF_COLUMN_NAME_USER_ID,
        LF_COLUMN_NAME_CONVERSATION,
        LF_COLUMN_NAME_MESSAGE
    ]
    df_small = df[cols]
    #df_small = df.reindex(columns=cols)
    log.info(f"DataFrame reduced to the following columns: {cols}")

    return df_small

ANALYZE_EXCEL = False
ANALYZE_LANGFUSE = True

if ANALYZE_EXCEL:
    log.info(f"Get data from Excel file '{messages_log_file}' ...")
    excel_df = pd.read_excel(messages_log_file)
    print(excel_df.head())
    number_of_unique_users = extract_some_metadata(excel_df, "User ID", "Conversation ID", "Date")

if ANALYZE_LANGFUSE:
    log.info("\n\n")
    latest_file, newest_timestamp = find_latest_traces_file()
    FETCH_FROM_LANGFUSE = True
    if FETCH_FROM_LANGFUSE:
        max_pages = 1
        #max_pages = 200  # stop after 200 pages
        if newest_timestamp is not None:
            log.info(f"Incremental fetch: only retrieving traces newer than {newest_timestamp} ...")
            lf_df = get_data_from_langfuse(max_pages, from_timestamp=newest_timestamp)
        else:
            log.info("No existing traces file found, fetching all traces from Langfuse ...")
            lf_df = get_data_from_langfuse(max_pages)
    else:
        if latest_file:
            log.info(f"Reading traces from '{latest_file}' only, but not from Langfuse host {lf_host} ...")
            lf_df = pd.read_csv(latest_file)
        else:
            log.error("No traces file found! Set FETCH_FROM_LANGFUSE = True to fetch from Langfuse.")
            exit(1)

    lf_df_small = reduce_data_frame(lf_df)

    # Drop all entries, which do not have a message
    lf_df_small = lf_df_small.dropna(subset=[LF_COLUMN_NAME_MESSAGE])

    # Drop test requests sent by user "68c3d92641aa5e1af33e9374"
    lf_df_small = lf_df_small[lf_df_small[LF_COLUMN_NAME_USER_ID] != "68c3d92641aa5e1af33e9374"]

    print(lf_df_small.head())
    number_of_unique_users = extract_some_metadata(lf_df_small, LF_COLUMN_NAME_USER_ID, LF_COLUMN_NAME_CONVERSATION, LF_COLUMN_NAME_DATE)

# TODO: Transform excel_df or lf_df_small into a source independent DataFrame

# Pick the user UUID you want to analyze
#user_id = "68c93b60279c6a9faf4683f3" # 441 messages and 11 conversations during the period Sept 11 to Oct 20
#user_id = "68c3d7a0599baa89eb48bc00" # 273 messages and 21 conversations during the period Sept 11 to Oct 20
#user_id = "68c3d7b1f2978f6b2a432235" # 257 messages and 23 conversations during the period Sept 11 to Oct 20
#user_id = "68c3d7ec41aa5e1af33e8d90" # 218 messages and 40 conversations during the period Sept 11 to Oct 20
#user_id = "68da33f9a8926c6d7d2e6009" # 195 messages and 14 conversations during the period Sept 11 to Oct 20
#user_id = "68c828cf66b3bfa761af5b77" # 188 messages and 47 conversations during the period Sept 11 to Oct 20
user_id = "691ae6fb9653e3d85b29067f" # Michael
#user_id = "68c3d813f2978f6b2a432e6f" # TODO: Which user is this?
#user_id = "68c3d92641aa5e1af33e9374" # Test User of Lena

print("\nChoose analysis function(s):\n")
print("1 - Ranking of users by number of conversations and messages sent")
print("2 - Conversations and messages per day")
print(f"3 - Conversations and messages of user '{user_id}' per day")
print("4 - Classify user messages by LLM")
print("5 - Show classification ranking by message volume")
print("\nEnter one or more numbers separated by commas (e.g. 1,3,5), or 'all' to run everything.\n")
raw_input = input("Your choice: ").strip().lower()

valid_choices = {"1", "2", "3", "4", "5"}
if raw_input == "all":
    choices = valid_choices
else:
    choices = {c.strip() for c in raw_input.split(",")}
    invalid = choices - valid_choices
    if invalid:
        log.error(f"Invalid choice(s): {', '.join(sorted(invalid))}. Please enter numbers between 1 and 5.")
        exit(1)

for choice in sorted(choices):
    log.info(f"\n--- Running function {choice} ---")

    if choice == "1":
        top_k = 20
        #top_k = number_of_unique_users
        if ANALYZE_EXCEL:
            ranking_by_conversations_and_messages_by_user(excel_df, top_k)
        if ANALYZE_LANGFUSE:
            ranking_by_conversations_and_messages_by_user(lf_df_small, top_k, LF_COLUMN_NAME_DATE, LF_COLUMN_NAME_USER_ID, LF_COLUMN_NAME_CONVERSATION)

    elif choice == "2":
        if ANALYZE_EXCEL:
            conversations_and_messages_per_day(excel_df)
        if ANALYZE_LANGFUSE:
            conversations_and_messages_per_day(lf_df_small, LF_COLUMN_NAME_DATE, LF_COLUMN_NAME_CONVERSATION)

    elif choice == "3":
        if ANALYZE_EXCEL:
            conversations_and_messages_of_user_per_day(excel_df, user_id)
        if ANALYZE_LANGFUSE:
            conversations_and_messages_of_user_per_day(lf_df_small, user_id, LF_COLUMN_NAME_DATE, LF_COLUMN_NAME_USER_ID, LF_COLUMN_NAME_CONVERSATION, LF_COLUMN_NAME_MESSAGE)

    elif choice == "4":
        if ANALYZE_EXCEL:
            batch_size = 3
            classify_messages_by_LLM(excel_df, messages_log_file, None, batch_size)
        if ANALYZE_LANGFUSE:
            batch_size = None
            classify_messages_by_LLM(lf_df_small, None, pickle_file_name, batch_size, LF_COLUMN_NAME_MESSAGE)

    elif choice == "5":
        excluded_user_ids = ["68c3d92641aa5e1af33e9374"]  # Test User of Lena
        if ANALYZE_EXCEL:
            display_classification_results(excel_df, excluded_user_ids)
        if ANALYZE_LANGFUSE:
            pickle_df = pd.read_pickle(pickle_file_name)
            display_classification_results(pickle_df, excluded_user_ids, LF_COLUMN_NAME_USER_ID)
