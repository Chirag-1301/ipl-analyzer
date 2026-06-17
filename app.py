import streamlit as st
import pandas as pd
import sqlite3
import os
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_sql_query_chain


# ==========================================
# PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="IPL Analytics Platform", page_icon="🏏", layout="wide")

# Secure API Key Handling via secrets.toml
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "YOUR_LOCAL_KEY")

db_path = "ipl_history.db"

@st.cache_resource
def get_db_connection():
    return sqlite3.connect(db_path, check_same_thread=False)

conn = get_db_connection()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🏏 IPL Platform")
page = st.sidebar.radio("Navigation", ["📊 Data Explorer", "🤖 AI Data Copilot"])
st.sidebar.divider()
st.sidebar.caption("Built with Streamlit & Gemini")

# ==========================================
# PAGE 1: FULLSCREEN DATA EXPLORER
# ==========================================
if page == "📊 Data Explorer":
    st.title("🏏 IPL Data Explorer")
    
    tab_match, tab_team, tab_player = st.tabs(["📊 Matches", "🛡️ Teams", "👤 Players"])
    
    # --- 1. THE MATCH JOURNEY ---
    with tab_match:
        st.markdown("**Select any match row** to drill down into the detailed scorecard.")
        
        # Load match data
        df_matches = pd.read_sql("SELECT match_id, season, date, team1, team2, winner, player_of_match FROM matches ORDER BY date DESC", conn)
        df_matches['season'] = df_matches['season'].astype(str)
        
        # Interactive selection table
        match_event = st.dataframe(
            df_matches, width="stretch", height=250, on_select="rerun", 
            selection_mode="single-row", hide_index=True
        )
        
        if len(match_event.selection.rows) > 0:
            selected_row_idx = match_event.selection.rows[0]
            selected_match_id = df_matches.iloc[selected_row_idx]['match_id']
            match_title = f"{df_matches.iloc[selected_row_idx]['team1']} vs {df_matches.iloc[selected_row_idx]['team2']}"
            match_winner = df_matches.iloc[selected_row_idx]['winner']
            
            st.divider()
            st.subheader(f"Scorecard: {match_title}")
            
            # --- MATCH HEADER SUMMARY ---
            summary_sql = f"""
                SELECT 
                    inning, 
                    SUM(total_runs) as runs, 
                    SUM(is_wicket) as wickets,
                    SUM(CASE WHEN wides = 0 AND noballs = 0 THEN 1 ELSE 0 END) as balls
                FROM deliveries 
                WHERE match_id = '{selected_match_id}' 
                GROUP BY inning
            """
            df_summary = pd.read_sql(summary_sql, conn)
            score_strings = [
                f"**Inning {r['inning']}:** {r['runs']}/{r['wickets']} ({r['balls'] // 6}.{r['balls'] % 6} Ov)" 
                for _, r in df_summary.iterrows()
            ]
            st.markdown(" | ".join(score_strings))
            st.success(f"🏆 **{match_winner}** won the match")
            
            # --- INNINGS TABS ---
            innings_list = df_summary['inning'].tolist()
            inning_tabs = st.tabs([f"Innings {i}" for i in innings_list])
            
            for idx, inning_num in enumerate(innings_list):
                with inning_tabs[idx]:
                    # --- BATTING SCORECARD ---
                    batting_sql = f"""
                        SELECT 
                            batter AS Batting, 
                            CASE WHEN MAX(player_out) IS NOT NULL THEN MAX(dismissal_kind) ELSE 'not out' END AS Status,
                            SUM(batter_runs) AS R,
                            SUM(CASE WHEN wides = 0 THEN 1 ELSE 0 END) AS B,
                            SUM(CASE WHEN batter_runs = 4 THEN 1 ELSE 0 END) AS "4s",
                            SUM(CASE WHEN batter_runs = 6 THEN 1 ELSE 0 END) AS "6s",
                            ROUND(
                                (SUM(batter_runs) * 100.0) /
                                NULLIF(SUM(CASE WHEN wides = 0 THEN 1 ELSE 0 END), 0),
                            2) AS "S/R"
                        FROM deliveries 
                        WHERE match_id = '{selected_match_id}' AND inning = {inning_num} 
                        GROUP BY batter
                        ORDER BY MIN(over), MIN(ball)
                    """
                    st.markdown("##### Batting")
                    df_batting = pd.read_sql(batting_sql, conn)
                    df_batting["S/R"] = df_batting["S/R"].fillna(0.0)
                    st.dataframe(df_batting, width="stretch", hide_index=True)

                    # --- BOWLING SCORECARD ---
                    bowling_sql = f"""
                        SELECT 
                            bowler AS Bowling,
                            (SUM(CASE WHEN wides = 0 AND noballs = 0 THEN 1 ELSE 0 END) / 6) || '.' ||
                            (SUM(CASE WHEN wides = 0 AND noballs = 0 THEN 1 ELSE 0 END) % 6) AS O,
                            SUM(total_runs) AS R,
                            SUM(CASE WHEN is_wicket = 1 AND dismissal_kind != 'run out' THEN 1 ELSE 0 END) AS W,
                            ROUND(
                                (SUM(total_runs) * 6.0) /
                                NULLIF(SUM(CASE WHEN wides = 0 AND noballs = 0 THEN 1 ELSE 0 END), 0),
                            2) AS ECON
                        FROM deliveries 
                        WHERE match_id = '{selected_match_id}' AND inning = {inning_num} 
                        GROUP BY bowler
                    """
                    st.markdown("##### Bowling")
                    st.dataframe(pd.read_sql(bowling_sql, conn), width="stretch", hide_index=True)

    # --- 2. THE TEAM JOURNEY ---
    with tab_team:
        st.markdown("**Select a team** to view their historical performance.")
        teams_sql = "SELECT DISTINCT team1 as Team FROM matches UNION SELECT DISTINCT team2 as Team FROM matches"
        df_all_teams = pd.read_sql(teams_sql, conn).dropna()
        team_event = st.dataframe(df_all_teams, width="stretch", height=250, on_select="rerun", selection_mode="single-row", hide_index=True)
        
        if len(team_event.selection.rows) > 0:
            selected_team = df_all_teams.iloc[team_event.selection.rows[0]]['Team']
            st.divider()
            st.subheader(f"Wins per Season: {selected_team}")
            stats_sql = f"""
                SELECT season, COUNT(winner) as Wins 
                FROM matches 
                WHERE winner = '{selected_team}' 
                GROUP BY season
                ORDER BY season DESC
            """
            stats = pd.read_sql(stats_sql, conn)
            stats['season'] = stats['season'].astype(str)
            st.bar_chart(stats.set_index("season"))

    # --- 3. THE PLAYER JOURNEY ---
    with tab_player:
        st.info("Player drill-down coming soon!")

# ==========================================
# PAGE 2: FULLSCREEN AI COPILOT
# ==========================================
elif page == "🤖 AI Data Copilot":
    st.title("🤖 AI Database Copilot")
    st.markdown("Ask anything about the IPL dataset. The AI will write the SQL, allow you to review it, and execute it against the database.")
    
    if "generated_sql" not in st.session_state: 
        st.session_state.generated_sql = ""

    user_question = st.chat_input("E.g., 'Which bowler has the most wickets?'")
    
    if user_question:
        with st.spinner("Analyzing database schema and writing SQL..."):
            db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
            raw = create_sql_query_chain(llm, db).invoke({"question": user_question})
            
            # Safely clean the string without causing EOL errors
            clean = raw.replace("```sql", "").replace("```", "").strip()
            if "SQLQuery:" in clean:
                clean = clean.split("SQLQuery:", 1)[1].strip()
                
            st.session_state.generated_sql = clean

    # Human-in-the-loop Execution
    if st.session_state.generated_sql:
        st.info("💡 **Review & Edit SQL**")
        edited_sql = st.text_area("Generated SQL", value=st.session_state.generated_sql, height=150, label_visibility="collapsed")
        
        if st.button("▶️ Execute Data Search", type="primary", use_container_width=True):
            with st.spinner("Running query..."):
                try:
                    result_df = pd.read_sql(edited_sql, conn)
                    st.success("Query executed successfully!")
                    st.dataframe(result_df, width="stretch", hide_index=True)
                except Exception as e: 
                    st.error(f"Database Error: {e}")
