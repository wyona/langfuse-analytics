import logging
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

DATE_ONLY = "DateOnly"

def conversations_and_messages_per_day(df: pd.DataFrame, dateColumnName: str = "Date", conversationIdColumnName: str = "Conversation ID"):
    """
    Display conversations and messages per day
    :param df: Pandas dataframe containing conversations and messages
    """
    log.info("\n\nDisplay conversations and messages per day ...")

    # Ensure 'Date' is a datetime
    df[dateColumnName] = pd.to_datetime(df[dateColumnName])

    # Extract only the date part (remove hours/min/sec)
    df[DATE_ONLY] = df[dateColumnName].dt.date

    # Count number of messages per day
    messages_per_day = df.groupby(DATE_ONLY).size()

    # Count number of unique conversations per day
    conversations_per_day = df.groupby(DATE_ONLY)[conversationIdColumnName].nunique()

    # Combine into a single DataFrame for plotting
    daily_stats = pd.DataFrame({
        "Messages": messages_per_day,
        "Conversations": conversations_per_day
    }).sort_index()

    log.info(daily_stats.head())  # sanity check

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(daily_stats.index, daily_stats["Messages"], label="Messages per day", marker="o")
    plt.plot(daily_stats.index, daily_stats["Conversations"], label="Conversations per day", marker="o")

    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.title("Messages and Conversations per Day")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def conversations_and_messages_of_user_per_day(df: pd.DataFrame, user_id: str, dateColumnName: str = "Date", userIdColumnName: str = "User ID", conversationIdColumnName: str = "Conversation ID", messageColumnName: str = "User Question"):
    """
    Display conversations and messages of a particular user per day
    :param df: Pandas dataframe containing conversations and messages
    :param user_id: User ID, e.g., "68c93b60279c6a9faf4683f3"
    :param userIdColumnName: Column name of user ID, e.g., "User ID" or "userId"
    :param messageColumnName: Column name of messages to display, e.g., "input.messages"
    """

    print(f"\n\nDisplay conversations and messages per day for user {user_id} ...\n\n")

    # Make sure 'Date' is parsed correctly
    #print(f"Date(s): {df[dateColumnName]}----")
    df[dateColumnName] = pd.to_datetime(df[dateColumnName])
    df[DATE_ONLY] = df[dateColumnName].dt.date

    # Filter for this user
    user_df = df[df[userIdColumnName] == user_id]

    #for index, row in user_df.iterrows():
    #    print(f"Row: {row.to_dict()}")

    user_messages = user_df[messageColumnName].values
    for msg in user_messages:
        print(f"MESSAGE:\n {msg}")

    # Count messages per day
    # TODO:  Consider to divide by 2, because the user request and the bot response each generate a trace / message
    messages_per_day = user_df.groupby(DATE_ONLY).size()
    # messages_per_day = user_df.groupby(DATE_ONLY).size() / 2

    # Count unique conversations per day
    conversations_per_day = user_df.groupby(DATE_ONLY)[conversationIdColumnName].nunique()

    # Combine into one DataFrame for plotting
    daily_stats = pd.DataFrame({
        "Messages": messages_per_day,
        "Conversations": conversations_per_day
    }).sort_index()

    print(daily_stats.head())  # sanity check

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_stats.index, daily_stats["Messages"], label="Messages per day", marker="o")
    plt.plot(daily_stats.index, daily_stats["Conversations"], label="Conversations per day", marker="o")

    plt.title(f"Messages and Conversations per Day for User {user_id}")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()