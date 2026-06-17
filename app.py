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
# SESSION STATE & LAYOUT TOGGLE
# ==========================================
# Set the IPL Copilot to be expanded by default
if "show_copilot" not in st.session_state:
    st.session_state.show_copilot = True

# Header area with Title and Copilot Toggle
header_col1, header_col2 = st.columns([8, 2])
with header_col1:
    st.title("🏏 IPL Data Explorer")
with header_col2:
    st.write("") # Vertical spacing
    # Dynamic arrow toggle
    btn_label = "▶ Collapse IPL Copilot" if st.session_state.show_copilot else "◀ Expand IPL Copilot"
    if st.button(btn_label, use_container_width=True):
        st.session_state.show_copilot = not st.session_state.show_copilot
        st.rerun()

st.divider()

# Dynamic Column Layout based on toggle state
if st.session_state.show_copilot:
    main_col, copilot_col = st.columns([7, 3], gap="large")
else:
    main_col = st.container()
    copilot_col = None

# ==========================================
# MAIN CONTENT: DATA EXPLORER
# ==========================================
with main_col:
    # Use st.radio with a unique key instead of st.tabs to prevent layout resets
    active_tab = st.radio(
        "Navigation", 
        ["📊 Matches", "🛡️ Teams", "👤 Players"], 
        horizontal=True, 
        label_visibility="collapsed", 
        key="main_nav_tab"
    )
    st.divider()
    
    # --- 1. THE MATCH JOURNEY ---
    if active_tab == "📊 Matches":
        st.markdown("**Select any match row** to drill down into the detailed scorecard.")
        
        # Load match data
        df_matches = pd.read_sql("SELECT match_id, season, date, team1, team2, winner, player_of_match FROM matches ORDER BY date DESC", conn)
        df_matches['season'] = df_matches['season'].astype(str)
        
        # Interactive selection table WITH A KEY
        match_event = st.dataframe(
            df_matches, width="stretch", height=250, on_select="rerun", 
            selection_mode="single-row", hide_index=True, key="match_table_selection"
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

    # --- 2. THE TEAM & SEASON JOURNEY ---
    elif active_tab == "🛡️ Teams":
        view_mode = st.radio("Select View:", ["🛡️ Team History", "🏆 Season Drilldown"], horizontal=True, key="team_view_mode")
        st.divider()

        if view_mode == "🛡️ Team History":
            st.markdown("**Select a team** to view their historical performance across all IPL seasons.")
            
            teams_sql = "SELECT DISTINCT team1 as Team FROM matches UNION SELECT DISTINCT team2 as Team FROM matches ORDER BY Team"
            df_all_teams = pd.read_sql(teams_sql, conn).dropna()
            selected_team = st.selectbox("Choose a Team:", df_all_teams['Team'].tolist(), key="selected_team_dropdown")
            
            if selected_team:
                st.subheader(f"All-Time Record: {selected_team}")
                team_history_sql = f"""
                    WITH TeamMatches AS (
                        SELECT season, team1 AS team, winner FROM matches WHERE team1 = '{selected_team}'
                        UNION ALL
                        SELECT season, team2 AS team, winner FROM matches WHERE team2 = '{selected_team}'
                    )
                    SELECT 
                        season AS Season,
                        COUNT(*) AS Played,
                        SUM(CASE WHEN winner = '{selected_team}' THEN 1 ELSE 0 END) AS Won,
                        SUM(CASE WHEN winner IS NOT NULL AND winner != '{selected_team}' THEN 1 ELSE 0 END) AS Lost,
                        SUM(CASE WHEN winner IS NULL THEN 1 ELSE 0 END) AS NoResult
                    FROM TeamMatches
                    GROUP BY season
                    ORDER BY CAST(SUBSTR(CAST(season AS TEXT), 1, 4) AS INTEGER) DESC
                """
                history_df = pd.read_sql(team_history_sql, conn)
                history_df['Season'] = history_df['Season'].astype(str)
                
                col_m1, col_m2 = st.columns([1, 2])
                with col_m1:
                    st.metric("Total Matches", history_df['Played'].sum())
                    st.metric("Total Wins", history_df['Won'].sum())
                    st.metric("Total Losses", history_df['Lost'].sum())
                    
                with col_m2:
                    st.dataframe(history_df, width="stretch", hide_index=True)
                    st.bar_chart(history_df.set_index("Season")['Won'])

        elif view_mode == "🏆 Season Drilldown":
            st.markdown("**Select a season** to view the points table and top performers.")
            
            seasons_sql = "SELECT DISTINCT season FROM matches ORDER BY CAST(SUBSTR(CAST(season AS TEXT), 1, 4) AS INTEGER) DESC"
            df_seasons = pd.read_sql(seasons_sql, conn)
            selected_season = st.selectbox("Choose a Season:", df_seasons['season'].tolist(), key="selected_season_dropdown")
            
            if selected_season:
                winner_sql = f"SELECT winner FROM matches WHERE season = '{selected_season}' ORDER BY date DESC LIMIT 1"
                season_winner_df = pd.read_sql(winner_sql, conn)
                season_winner = season_winner_df.iloc[0]['winner'] if not season_winner_df.empty else "N/A"
                
                orange_cap_sql = f"""
                    SELECT d.batter, SUM(d.batter_runs) as runs
                    FROM deliveries d
                    JOIN matches m ON d.match_id = m.match_id
                    WHERE m.season = '{selected_season}'
                    GROUP BY d.batter
                    ORDER BY runs DESC LIMIT 1
                """
                orange_cap = pd.read_sql(orange_cap_sql, conn)
                
                purple_cap_sql = f"""
                    SELECT d.bowler, SUM(CASE WHEN d.is_wicket = 1 AND d.dismissal_kind NOT IN ('run out', 'retired hurt', 'obstructing the field') THEN 1 ELSE 0 END) as wickets
                    FROM deliveries d
                    JOIN matches m ON d.match_id = m.match_id
                    WHERE m.season = '{selected_season}'
                    GROUP BY d.bowler
                    ORDER BY wickets DESC LIMIT 1
                """
                purple_cap = pd.read_sql(purple_cap_sql, conn)
                
                st.subheader(f"IPL {selected_season} Highlights")
                c1, c2, c3 = st.columns(3)
                
                c1.info(f"🏆 **Champions:**\n\n{season_winner}")
                
                if not orange_cap.empty:
                    c2.markdown(f"""
                    <div style="background-color: rgba(255, 140, 0, 0.15); border: 1px solid rgba(255, 140, 0, 0.4); padding: 1rem; border-radius: 0.5rem; height: 100%;">
                        <span style="color: #FF8C00; font-weight: bold;">🏏 Orange Cap:</span><br><br>{orange_cap.iloc[0]['batter']} ({orange_cap.iloc[0]['runs']} Runs)
                    </div>
                    """, unsafe_allow_html=True)
                
                if not purple_cap.empty:
                    c3.markdown(f"""
                    <div style="background-color: rgba(160, 32, 240, 0.15); border: 1px solid rgba(160, 32, 240, 0.4); padding: 1rem; border-radius: 0.5rem; height: 100%;">
                        <span style="color: #B472FF; font-weight: bold;">🎯 Purple Cap:</span><br><br>{purple_cap.iloc[0]['bowler']} ({purple_cap.iloc[0]['wickets']} Wickets)
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                st.subheader(f"Points Table Approximation ({selected_season})")
                points_table_sql = f"""
                    WITH TeamMatches AS (
                        SELECT team1 AS team, winner FROM matches WHERE season = '{selected_season}'
                        UNION ALL
                        SELECT team2 AS team, winner FROM matches WHERE season = '{selected_season}'
                    )
                    SELECT 
                        team AS Team,
                        COUNT(*) AS Played,
                        SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) AS Won,
                        SUM(CASE WHEN winner IS NOT NULL AND winner != team THEN 1 ELSE 0 END) AS Lost,
                        SUM(CASE WHEN winner IS NULL THEN 1 ELSE 0 END) AS NR,
                        (SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) * 2) + SUM(CASE WHEN winner IS NULL THEN 1 ELSE 0 END) AS Points
                    FROM TeamMatches
                    GROUP BY team
                    ORDER BY Points DESC, Won DESC
                """
                pts_df = pd.read_sql(points_table_sql, conn)
                pts_df.insert(0, 'Pos', range(1, 1 + len(pts_df))) 
                st.dataframe(pts_df, width="stretch", hide_index=True)

    # --- 3. THE PLAYER JOURNEY ---
    elif active_tab == "👤 Players":
        player_category = st.radio("Select Category:", ["🏏 Top Batters", "🎯 Top Bowlers", "🧤 Top Fielders"], horizontal=True, key="player_category_radio")
        st.divider()

        if player_category == "🏏 Top Batters":
            st.markdown("**All-Time Batting Leaderboard (Ranked by Total Score, Min. 10 Matches)**")
            batters_sql = """
                WITH POM_Counts AS (
                    SELECT player_of_match AS player_name, COUNT(*) AS pom_awards
                    FROM matches
                    WHERE player_of_match IS NOT NULL
                    GROUP BY player_of_match
                ),
                Match_Counts AS (
                    SELECT player, COUNT(DISTINCT match_id) AS matches_played
                    FROM (
                        SELECT batter AS player, match_id FROM deliveries
                        UNION SELECT non_striker AS player, match_id FROM deliveries
                        UNION SELECT bowler AS player, match_id FROM deliveries
                    )
                    WHERE player IS NOT NULL
                    GROUP BY player
                ),
                Batter_Stats AS (
                    SELECT 
                        batter AS Player,
                        SUM(batter_runs) AS Runs,
                        SUM(CASE WHEN batter_runs = 4 THEN 1 ELSE 0 END) AS `4s`,
                        SUM(CASE WHEN batter_runs = 6 THEN 1 ELSE 0 END) AS `6s`,
                        ROUND((SUM(batter_runs) * 100.0) / NULLIF(SUM(CASE WHEN wides = 0 THEN 1 ELSE 0 END), 0), 1) AS `Strike Rate`,
                        (SUM(batter_runs) + 
                         SUM(CASE WHEN batter_runs = 4 THEN 1 ELSE 0 END) + 
                         SUM(CASE WHEN batter_runs = 6 THEN 2 ELSE 0 END)) AS base_score
                    FROM deliveries 
                    GROUP BY batter
                )
                SELECT 
                    b.Player,
                    m.matches_played AS `Matches`,
                    b.Runs,
                    b.`4s`,
                    b.`6s`,
                    b.`Strike Rate`,
                    COALESCE(p.pom_awards, 0) AS `POM Awards`,
                    (b.base_score + (COALESCE(p.pom_awards, 0) * 50) + (m.matches_played * 5)) AS `Total Score`,
                    ROUND(CAST((b.base_score + (COALESCE(p.pom_awards, 0) * 50) + (m.matches_played * 5)) AS FLOAT) / m.matches_played, 2) AS `Score Per Match`
                FROM Batter_Stats b
                LEFT JOIN POM_Counts p ON b.Player = p.player_name
                JOIN Match_Counts m ON b.Player = m.player
                WHERE m.matches_played >= 10
                ORDER BY `Total Score` DESC
                LIMIT 500
            """
            df_batters = pd.read_sql(batters_sql, conn)
            max_pts = df_batters['Total Score'].max()
            df_batters['Overall Rating'] = (df_batters['Total Score'] / max_pts * 100).round(1)
            
            st.dataframe(
                df_batters,
                column_config={"Overall Rating": st.column_config.ProgressColumn("Overall Rating", format="%.1f / 100", min_value=0, max_value=100)},
                hide_index=True, use_container_width=True
            )

        elif player_category == "🎯 Top Bowlers":
            st.markdown("**All-Time Bowling Leaderboard (Ranked by Total Score, Min. 10 Matches)**")
            bowlers_sql = """
                WITH POM_Counts AS (
                    SELECT player_of_match AS player_name, COUNT(*) AS pom_awards
                    FROM matches
                    WHERE player_of_match IS NOT NULL
                    GROUP BY player_of_match
                ),
                Match_Counts AS (
                    SELECT player, COUNT(DISTINCT match_id) AS matches_played
                    FROM (
                        SELECT batter AS player, match_id FROM deliveries
                        UNION SELECT non_striker AS player, match_id FROM deliveries
                        UNION SELECT bowler AS player, match_id FROM deliveries
                    )
                    WHERE player IS NOT NULL
                    GROUP BY player
                ),
                Bowler_Stats AS (
                    SELECT 
                        bowler AS Player,
                        SUM(CASE WHEN is_wicket = 1 AND dismissal_kind NOT IN ('run out', 'retired hurt', 'obstructing the field') THEN 1 ELSE 0 END) AS Wickets,
                        SUM(CASE WHEN wides = 0 AND noballs = 0 AND batter_runs = 0 AND extra_runs = 0 THEN 1 ELSE 0 END) AS `Dot Balls`,
                        ROUND((SUM(total_runs) * 6.0) / NULLIF(SUM(CASE WHEN wides = 0 AND noballs = 0 THEN 1 ELSE 0 END), 0), 2) AS `Economy`,
                        (SUM(CASE WHEN is_wicket = 1 AND dismissal_kind NOT IN ('run out', 'retired hurt', 'obstructing the field') THEN 1 ELSE 0 END) * 25 + 
                         SUM(CASE WHEN wides = 0 AND noballs = 0 AND batter_runs = 0 AND extra_runs = 0 THEN 1 ELSE 0 END)) AS base_score
                    FROM deliveries 
                    GROUP BY bowler
                )
                SELECT 
                    b.Player,
                    m.matches_played AS `Matches`,
                    b.Wickets,
                    b.`Dot Balls`,
                    b.`Economy`,
                    COALESCE(p.pom_awards, 0) AS `POM Awards`,
                    (b.base_score + (COALESCE(p.pom_awards, 0) * 50) + (m.matches_played * 5)) AS `Total Score`,
                    ROUND(CAST((b.base_score + (COALESCE(p.pom_awards, 0) * 50) + (m.matches_played * 5)) AS FLOAT) / m.matches_played, 2) AS `Score Per Match`
                FROM Bowler_Stats b
                LEFT JOIN POM_Counts p ON b.Player = p.player_name
                JOIN Match_Counts m ON b.Player = m.player
                WHERE m.matches_played >= 10
                ORDER BY `Total Score` DESC
                LIMIT 500
            """
            df_bowlers = pd.read_sql(bowlers_sql, conn)
            max_pts = df_bowlers['Total Score'].max()
            df_bowlers['Overall Rating'] = (df_bowlers['Total Score'] / max_pts * 100).round(1)
            
            st.dataframe(
                df_bowlers,
                column_config={"Overall Rating": st.column_config.ProgressColumn("Overall Rating", format="%.1f / 100", min_value=0, max_value=100)},
                hide_index=True, use_container_width=True
            )

        elif player_category == "🧤 Top Fielders":
            st.markdown("**All-Time Fielding Leaderboard (Ranked by Total Score, Min. 10 Matches)**")
            fielders_sql = """
                WITH POM_Counts AS (
                    SELECT player_of_match AS player_name, COUNT(*) AS pom_awards
                    FROM matches
                    WHERE player_of_match IS NOT NULL
                    GROUP BY player_of_match
                ),
                Match_Counts AS (
                    SELECT player, COUNT(DISTINCT match_id) AS matches_played
                    FROM (
                        SELECT batter AS player, match_id FROM deliveries
                        UNION SELECT non_striker AS player, match_id FROM deliveries
                        UNION SELECT bowler AS player, match_id FROM deliveries
                    )
                    WHERE player IS NOT NULL
                    GROUP BY player
                ),
                Fielder_Stats AS (
                    SELECT 
                        fielder AS Player,
                        SUM(CASE WHEN dismissal_kind = 'caught' THEN 1 ELSE 0 END) AS Catches,
                        SUM(CASE WHEN dismissal_kind IN ('run out', 'stumped') THEN 1 ELSE 0 END) AS RO_Stump,
                        (SUM(CASE WHEN dismissal_kind = 'caught' THEN 1 ELSE 0 END) * 8 + 
                         SUM(CASE WHEN dismissal_kind IN ('run out', 'stumped') THEN 1 ELSE 0 END) * 12) AS base_score
                    FROM deliveries 
                    WHERE is_wicket = 1 AND fielder IS NOT NULL
                    GROUP BY fielder
                )
                SELECT 
                    f.Player,
                    m.matches_played AS `Matches`,
                    f.Catches,
                    f.RO_Stump,
                    COALESCE(p.pom_awards, 0) AS `POM Awards`,
                    (f.base_score + (COALESCE(p.pom_awards, 0) * 50) + (m.matches_played * 5)) AS `Total Score`,
                    ROUND(CAST((f.base_score + (COALESCE(p.pom_awards, 0) * 50) + (m.matches_played * 5)) AS FLOAT) / m.matches_played, 2) AS `Score Per Match`
                FROM Fielder_Stats f
                LEFT JOIN POM_Counts p ON f.Player = p.player_name
                JOIN Match_Counts m ON f.Player = m.player
                WHERE m.matches_played >= 10
                ORDER BY `Total Score` DESC
                LIMIT 500
            """
            try:
                df_fielders = pd.read_sql(fielders_sql, conn)
                max_pts = df_fielders['Total Score'].max()
                df_fielders['Overall Rating'] = (df_fielders['Total Score'] / max_pts * 100).round(1)
                
                st.dataframe(
                    df_fielders,
                    column_config={"Overall Rating": st.column_config.ProgressColumn("Overall Rating", format="%.1f / 100", min_value=0, max_value=100)},
                    hide_index=True, use_container_width=True
                )
            except Exception as e:
                st.error("Column error: Ensure your `deliveries` table has the `fielder` column extracted from the JSON!")

# ==========================================
# RIGHT COLUMN: IPL COPILOT
# ==========================================
if copilot_col is not None:
    with copilot_col:
        with st.container(border=True):
            st.markdown("### 🤖 <span style='color: #4B90FF'>IPL Copilot</span>", unsafe_allow_html=True)
            st.caption("Ask anything about the IPL dataset. The AI will write the SQL, allow you to review it, and execute it.")
            
            if "generated_sql" not in st.session_state: 
                st.session_state.generated_sql = ""

            user_question = st.chat_input("E.g., 'Which bowler has the most wickets?'")
            
            if user_question:
                with st.spinner("Writing SQL..."):
                    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
                    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
                    raw = create_sql_query_chain(llm, db).invoke({"question": user_question})
                    
                    clean = raw.replace("```sql", "").replace("```", "").strip()
                    if "SQLQuery:" in clean:
                        clean = clean.split("SQLQuery:", 1)[1].strip()
                        
                    st.session_state.generated_sql = clean

            if st.session_state.generated_sql:
                st.info("💡 **Review & Edit SQL**")
                edited_sql = st.text_area("Generated SQL", value=st.session_state.generated_sql, height=150, label_visibility="collapsed", key="sql_editor")
                
                if st.button("▶️ Execute Data Search", type="primary", use_container_width=True, key="execute_sql_btn"):
                    with st.spinner("Running query..."):
                        try:
                            result_df = pd.read_sql(edited_sql, conn)
                            st.success("Success!")
                            st.dataframe(result_df, width="stretch", hide_index=True)
                        except Exception as e: 
                            st.error(f"Database Error: {e}")
