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

TRACES_FILE = "langfuse_traces.csv"

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
        oldest_date = df[dateColumnName].min().date()
        youngest_date = df[dateColumnName].max().date()
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

def get_data_from_langfuse(max_pages: int):
    """
    Get data from Langfuse
    :param max_pages: Maximum number of pages to fetch
    :return: Dataframe containing conversations and messages
    """
    load_dotenv()
    lf_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    log.debug(f"LF_public_key: {lf_public_key}")
    lf_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    log.debug(f"LF_secret_key: {lf_secret_key}")
    lf_host = os.getenv("LANGFUSE_HOST")
    log.info(f"LF_host: {lf_host}")
    langfuse = Langfuse(
        public_key=lf_public_key,
        secret_key=lf_secret_key,
        host=lf_host,
        timeout=20,
    )
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

        response = langfuse.api.trace.list(limit=limit_per_page, page=page)
        # One can filter by tags, but excluding traces with certain tags / negative tagging is not possible
        #response = langfuse.api.trace.list(tags="litellm-internal-health-check", limit=limit_per_page, page=page)
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
    log.info(f"Number of traces with non-empty '{LF_COLUMN_NAME_USER_ID}' retrieved: {len(records)}")

    if not records:
        log.warning("No traces with non-empty '{LF_COLUMN_NAME_USER_ID}' retrieved!")
        lf_df = pd.DataFrame(columns=["id", LF_COLUMN_NAME_DATE, LF_COLUMN_NAME_USER_ID, LF_COLUMN_NAME_CONVERSATION])
    else:
        lf_df = pd.json_normalize(records)
        lf_df.to_csv(TRACES_FILE , index=False)
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
    if False:
        max_pages = 10
        #max_pages = 200  # stop after 200 pages
        lf_df = get_data_from_langfuse(max_pages)
    else:
        lf_df = pd.read_csv(TRACES_FILE )

    lf_df_small = reduce_data_frame(lf_df)

    # Drop all entries, which do not have a message
    lf_df_small = lf_df_small.dropna(subset=[LF_COLUMN_NAME_MESSAGE])

    # Drop test requests sent by user "68c3d92641aa5e1af33e9374"
    lf_df_small = lf_df_small[lf_df_small[LF_COLUMN_NAME_USER_ID] != "68c3d92641aa5e1af33e9374"]

    print(lf_df_small.head())
    number_of_unique_users = extract_some_metadata(lf_df_small, LF_COLUMN_NAME_USER_ID, LF_COLUMN_NAME_CONVERSATION, LF_COLUMN_NAME_DATE)

# TODO: Transform excel_df or lf_df_small into a source independent DataFrame

# Run various functions to analyze conversations and messages
if True:
    top_k = 20
    #top_k = number_of_unique_users
    if ANALYZE_EXCEL:
        ranking_by_conversations_and_messages_by_user(excel_df, top_k)

    if ANALYZE_LANGFUSE:
        ranking_by_conversations_and_messages_by_user(lf_df_small, top_k, LF_COLUMN_NAME_DATE, LF_COLUMN_NAME_USER_ID, LF_COLUMN_NAME_CONVERSATION)

if True:
    if ANALYZE_EXCEL:
        conversations_and_messages_per_day(excel_df)
    if ANALYZE_LANGFUSE:
        conversations_and_messages_per_day(lf_df_small, LF_COLUMN_NAME_DATE, LF_COLUMN_NAME_CONVERSATION)

if True:
    # Pick the user UUID you want to analyze
    #user_id = "68c93b60279c6a9faf4683f3" # 441 messages and 11 conversations during the period Sept 11 to Oct 20
    #user_id = "68c3d7a0599baa89eb48bc00" # 273 messages and 21 conversations during the period Sept 11 to Oct 20
    #user_id = "68c3d7b1f2978f6b2a432235" # 257 messages and 23 conversations during the period Sept 11 to Oct 20
    #user_id = "68c3d7ec41aa5e1af33e8d90" # 218 messages and 40 conversations during the period Sept 11 to Oct 20
    #user_id = "68da33f9a8926c6d7d2e6009" # 195 messages and 14 conversations during the period Sept 11 to Oct 20
    #user_id = "68c828cf66b3bfa761af5b77" # 188 messages and 47 conversations during the period Sept 11 to Oct 20
    user_id = "691ae6fb9653e3d85b29067f" # Michael
    #user_id = "68c3d813f2978f6b2a432e6f" # TODO
    #user_id = "68c3d92641aa5e1af33e9374" # Test User of Lena
    if ANALYZE_EXCEL:
        conversations_and_messages_of_user_per_day(excel_df, user_id)
    if ANALYZE_LANGFUSE:
        conversations_and_messages_of_user_per_day(lf_df_small, user_id, LF_COLUMN_NAME_DATE, LF_COLUMN_NAME_USER_ID, LF_COLUMN_NAME_CONVERSATION, LF_COLUMN_NAME_MESSAGE)

if False:
    if ANALYZE_EXCEL:
        # batch_size = 10
        batch_size = 3
        classify_messages_by_LLM(excel_df, messages_log_file, None, batch_size)
    if ANALYZE_LANGFUSE:
        #batch_size = 3
        batch_size = None
        classify_messages_by_LLM(lf_df_small, None, pickle_file_name, batch_size, LF_COLUMN_NAME_MESSAGE)

if False:
    #excluded_user_ids = ["68c93b60279c6a9faf4683f3", "68c3d7a0599baa89eb48bc00", "68c3d7b1f2978f6b2a432235", "68c3d7ec41aa5e1af33e8d90", "68da33f9a8926c6d7d2e6009", "68c828cf66b3bfa761af5b77"]
    #excluded_user_ids = ["68c93b60279c6a9faf4683f3"]
    excluded_user_ids = ["68c3d92641aa5e1af33e9374"] # Test User of Lena
    if ANALYZE_EXCEL:
        display_classification_results(excel_df, excluded_user_ids)
        #display_classification_results(excel_df)
    if ANALYZE_LANGFUSE:
        pickle_df = pd.read_pickle(pickle_file_name)
        display_classification_results(pickle_df, excluded_user_ids, LF_COLUMN_NAME_USER_ID)
