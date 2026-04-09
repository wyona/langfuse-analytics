import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

MESSAGE_TRACES = "Messages_Traces"
CONVERSATION_SESSIONS = "Conversations_Sessions"

def ranking_by_conversations_and_messages_by_user(df: pd.DataFrame, top_k: int = 20, dateColumnName: str = "Date", userIdColumnName: str = "User ID", conversationIdColumnName: str = "Conversation ID"):
    """
    Rank by conversations and messages per user, which user had the most conversations, which user sent most messages
    :param df: Pandas dataframe containing conversations and messages
    :param top_k: Output / display top k ranking results
    :param dateColumnName: Column name of date, e.g. "Date" or "timestamp"
    :param userIdColumnName: Column name of user ID, e.g. "User ID" or "userId"
    :param conversationIdColumnName: Column name of conversation ID, e.g. "Conversation ID" or "sessionId"
    """
    log.info("\n\nRanking by conversations and messages by user ...")

    if df.empty:
        log.warning("DataFrame is empty, therefore no conversations or messages to rank ...")
        return

    df.loc[:, dateColumnName] = pd.to_datetime(df[dateColumnName])
    oldest_date = df[dateColumnName].min().date()
    youngest_date = df[dateColumnName].max().date()
    log.debug(f"Oldest date: {oldest_date}")
    log.debug(f"Youngest date: {youngest_date}")

    # Do not truncate the output display, but show all users
    # pd.set_option("display.max_rows", None)

    # --- Ranking 1: by number of conversations per user ---
    # Each user may have multiple rows per conversation,
    # so we count unique Conversation IDs per User.
    conversation_count = (
        df.groupby(userIdColumnName)[conversationIdColumnName]
        .nunique()
        .sort_values(ascending=False)
    )

    # --- Ranking 2: by number of messages per user ---
    # Each row is a message, so just count rows per user
    # TODO:  Consider to divide by 2, because the user request and the bot response each generate a trace / message
    message_count = (
        df[userIdColumnName]
        .value_counts()
        .sort_values(ascending=False)
    )

    # --- Combine both into one DataFrame for clarity ---
    ranking = pd.DataFrame({
        CONVERSATION_SESSIONS: conversation_count,
        MESSAGE_TRACES: message_count
    }).fillna(0).astype(int)

    # log.info(f"\n\nTop {top_k} users by conversations:")
    # ranking_sorted = ranking.sort_values("Conversations", ascending=False)
    # print(ranking_sorted.head(top_k))

    log.info(f"\n\nTop {top_k} users by messages / traces:")
    ranking_sorted = ranking.sort_values(MESSAGE_TRACES, ascending=False)
    # print(f"Shape: {ranking_sorted.shape}")
    print(ranking.sort_values(MESSAGE_TRACES, ascending=False).head(top_k))

    # Dump ranking as CSV, such that one can import it manually into for example Power BI
    ranking_sorted.to_csv(f"user_ranking_{oldest_date}_to_{youngest_date}.csv", index=True)

    mean_messages = ranking[MESSAGE_TRACES].mean()
    median_messages = ranking[MESSAGE_TRACES].median()
    print(f"Mean messages / traces per user: {mean_messages:.2f}")
    print(f"Median messages / traces per user: {median_messages}")

    mean_conversations = ranking[CONVERSATION_SESSIONS].mean()
    median_conversations = ranking[CONVERSATION_SESSIONS].median()
    print(f"Mean conversations / sessions per user: {mean_conversations:.2f}")
    print(f"Median conversations / sessions per user: {median_conversations}")

    plt.figure(figsize=(14, 6))
    # Create x-axis positions starting from 1
    start_x_pos = 0

    #x_positions = range(start_x_pos, start_x_pos + len(ranking_sorted))
    #plt.bar(x_positions, ranking_sorted[MESSAGE_TRACES], label="Messages / Traces", alpha=0.7)
    #plt.bar(x_positions, ranking_sorted[CONVERSATION_SESSIONS], label="Conversations / Sessions", alpha=0.7)

    x_positions = np.arange(len(ranking_sorted)) + start_x_pos
    width = 0.4
    plt.bar(x_positions - width / 2,
            ranking_sorted[MESSAGE_TRACES],
            width=width,
            label="Messages / Traces",
            alpha=0.7)
    plt.bar(x_positions + width / 2,
            ranking_sorted[CONVERSATION_SESSIONS],
            width=width,
            label="Conversations / Sessions",
            alpha=0.7)

    plt.xlabel(f"User Index ({ranking_sorted.shape[0]} unique Users)")
    plt.ylabel("Count")
    plt.title(f"Conversations/Sessions and Messages/Traces per User during the period {oldest_date} to {youngest_date}")
    plt.xticks([])  # remove UUIDs
    plt.legend()
    plt.tight_layout()
    plt.show()
