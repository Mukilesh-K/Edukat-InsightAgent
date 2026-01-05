import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Chatbot Cost Analytics",
    page_icon="💰",
    layout="wide"
)

# Database Connection
def get_db_connection():
    try:
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")
        database = os.getenv("database")
        
        DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None

engine = get_db_connection()

# Sidebar Filters
st.sidebar.header("Filters")
today = datetime.now().date()
start_date = st.sidebar.date_input("Start Date", today - timedelta(days=30))
end_date = st.sidebar.date_input("End Date", today)

if start_date > end_date:
    st.sidebar.error("Error: Start date must be before end date.")

# Main Title
st.title("Chatbot Cost & Usage Analytics")

if engine:
    # ---------------------------------------------------------
    # 1. Fetch Data
    # ---------------------------------------------------------
    params = {
        "start_date": start_date,
        "end_date": end_date + timedelta(days=1)
    }
    
    # Query for Metrics and Trend
    # Joining with chatbot_master to get user_type if needed, 
    # but primarily analyzing the user queries.
    base_query = """
    SELECT 
        q.created_at,
        q.token_usage,
        q.token_cost,
        q.intent_type,
        q.query,
        q.response,
        m.user_type,
        m.session_id
    FROM chatbot_user_query q
    JOIN chatbot_master m ON q.chatbot_master_id = m.id
    WHERE q.created_at BETWEEN :start_date AND :end_date
    """
    
    try:
        df = pd.read_sql_query(text(base_query), engine, params=params)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # ---------------------------------------------------------
        # 2. Key Metrics
        # ---------------------------------------------------------
        total_queries = len(df)
        total_tokens = df['token_usage'].sum()
        total_cost = df['token_cost'].sum()
        avg_cost_per_query = total_cost / total_queries if total_queries > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", f"{total_queries:,}")
        col2.metric("Total Tokens", f"{total_tokens:,}")
        col3.metric("Total Cost", f"${total_cost:,.4f}")
        col4.metric("Avg Cost/Query", f"${avg_cost_per_query:,.4f}")

        st.divider()

        # ---------------------------------------------------------
        # 3. Trends (Charts)
        # ---------------------------------------------------------
        col_chart1, col_chart2 = st.columns(2)

        # A. Usage & Cost Over Time
        # Group by date
        daily_df = df.groupby('date')[['token_usage', 'token_cost']].sum().reset_index()

        with col_chart1:
            st.subheader("Token Usage & Cost Trend")
            
            fig_trend = go.Figure()
            # Bar for Usage
            fig_trend.add_trace(
                go.Bar(
                    x=daily_df['date'],
                    y=daily_df['token_usage'],
                    name="Token Usage",
                    marker_color='lightblue',
                    yaxis='y'
                )
            )
            # Line for Cost
            fig_trend.add_trace(
                go.Scatter(
                    x=daily_df['date'],
                    y=daily_df['token_cost'],
                    name="Cost ($)",
                    mode='lines+markers',
                    line=dict(color='orange', width=3),
                    yaxis='y2'
                )
            )

            fig_trend.update_layout(
                xaxis_title="Date",
                yaxis=dict(title="Tokens", side='left', showgrid=False),
                yaxis2=dict(title="Cost ($)", side='right', overlaying='y', showgrid=False),
                legend=dict(x=0, y=1.2, orientation='h'),
                hovermode="x unified"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # B. Intent Distribution
        with col_chart2:
            st.subheader("Intent Distribution")
            intent_counts = df['intent_type'].value_counts().reset_index()
            intent_counts.columns = ['intent', 'count']
            
            if not intent_counts.empty:
                fig_pie = px.pie(
                    intent_counts, 
                    values='count', 
                    names='intent',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No intent data available.")

        st.divider()

        # ---------------------------------------------------------
        # 4. Detailed Data View
        # ---------------------------------------------------------
        st.subheader("Detailed Interaction Log")
        
        # Display relevant columns
        display_df = df[[
            'created_at', 'user_type', 'intent_type', 
            'query', 'response', 'token_usage', 'token_cost'
        ]].sort_values(by='created_at', ascending=False)
        
        st.dataframe(
            display_df, 
            use_container_width=True,
            column_config={
                "created_at": st.column_config.DatetimeColumn("Timestamp", format="D MMM, HH:mm"),
                "token_cost": st.column_config.NumberColumn("Cost ($)", format="$%.4f")
            }
        )
    else:
        st.info("No data found for the selected date range.")

else:
    st.warning("Please verify your database connection credentials in .env file.")
