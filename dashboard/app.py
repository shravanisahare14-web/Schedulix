import sys
import datetime
import heapq
import csv
import io
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# =====================================================================
# 1. CORE GRAPH DATA STRUCTURES
# =====================================================================
class Task:
    def __init__(self, task_id, name, duration, skill, due_day, profit=100, dep=None):
        self.id = str(task_id)
        self.name = str(name)
        self.duration = max(1, int(duration))
        self.skill = str(skill)
        self.due_day = int(due_day)
        self.profit = int(profit)  
        self.dep = None if dep in ["None", "", None] else str(dep)       
        self.start_day = None
        self.finish_day = None
        self.lateness = 0
        self.is_missed = False
        self.net_score = 0

    def calculate_metrics(self, actual_start):
        self.start_day = actual_start
        self.finish_day = actual_start + self.duration
        self.lateness = max(0, self.finish_day - self.due_day)
        self.is_missed = self.finish_day > self.due_day
        self.net_score = max(0, self.profit - (self.lateness * 15))


# =====================================================================
# 2. OPTIMIZATION CORE LIFECYCLES
# =====================================================================
class SchedulixEngine:
    @staticmethod
    def run_greedy_heap_scheduler(tasks_input, w_scale=100, r_drop=0):
        task_pool = [Task(t.id, t.name, t.duration, t.skill, t.due_day, t.profit, t.dep) for t in tasks_input]
        task_map = {t.id: t for t in task_pool}
        adj = {t.id: [] for t in task_pool}
        in_degree = {t.id: 0 for t in task_pool}
        
        for t in task_pool:
            if t.dep and t.dep in task_map:
                adj[t.dep].append(t.id)
                in_degree[t.id] += 1
        
        efficiency_modifier = 1.0 / (1.0 - (r_drop / 100.0)) if r_drop < 100 else 1.0
        duration_scale = (w_scale / 100.0) * efficiency_modifier
        for t in task_pool:
            t.duration = max(1, round(t.duration * duration_scale))

        ready_heap = []
        for t_id, degree in in_degree.items():
            if degree == 0:
                heapq.heappush(ready_heap, (task_map[t_id].due_day, t_id))
        
        ordered_sequence = []
        skill_busy_until = {}
        task_finish_day = {}
        total_profit = 0
        missed_count = 0
        
        while ready_heap:
            _, curr_id = heapq.heappop(ready_heap)
            t = task_map[curr_id]
            dep_finish = task_finish_day.get(t.dep, 0) if t.dep else 0
            resource_checkpoint = skill_busy_until.get(t.skill, 0)
            calculated_start = max(dep_finish, resource_checkpoint)
            t.calculate_metrics(calculated_start)
            
            task_finish_day[curr_id] = t.finish_day
            skill_busy_until[t.skill] = t.finish_day
            total_profit += t.net_score
            if t.is_missed:
                missed_count += 1
            ordered_sequence.append(t)
            
            for neighbor in adj[curr_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(ready_heap, (task_map[neighbor].due_day, neighbor))
                    
        return ordered_sequence, total_profit, missed_count

    @staticmethod
    def run_cpsat_solver(tasks_input, w_scale=100):
        if not ORTOOLS_AVAILABLE or len(tasks_input) == 0:
            return SchedulixEngine.run_greedy_heap_scheduler(tasks_input, w_scale, 0)
            
        task_pool = [Task(t.id, t.name, t.duration, t.skill, t.due_day, t.profit, t.dep) for t in tasks_input]
        model = cp_model.CpModel()
        horizon = 500  
        
        duration_scale = w_scale / 100.0
        task_starts, task_ends, task_intervals, net_score_vars = {}, {}, {}, []
        
        for t in task_pool:
            t.duration = max(1, round(t.duration * duration_scale))
            start = model.NewIntVar(0, horizon, f"start_{t.id}")
            end = model.NewIntVar(0, horizon, f"end_{t.id}")
            interval = model.NewIntervalVar(start, t.duration, end, f"int_{t.id}")
            
            task_starts[t.id], task_ends[t.id], task_intervals[t.id] = start, end, interval
            lateness = model.NewIntVar(0, horizon, f"late_{t.id}")
            model.AddMaxEquality(lateness, [0, end - t.due_day])
            
            raw_score = model.NewIntVar(-horizon * 15, t.profit, f"raw_score_{t.id}")
            model.Add(raw_score == t.profit - (lateness * 15))
            
            net_score = model.NewIntVar(0, t.profit, f"score_{t.id}")
            model.AddMaxEquality(net_score, [0, raw_score])
            net_score_vars.append(net_score)

        for t in task_pool:
            if t.dep and t.dep in task_starts:
                model.Add(task_starts[t.id] >= task_ends[t.dep])
                
        skill_groups = {}
        for t in task_pool:
            skill_groups.setdefault(t.skill, []).append(task_intervals[t.id])
        for skill, intervals in skill_groups.items():
            if len(intervals) > 1:
                model.AddNoOverlap(intervals)
                
        model.Maximize(sum(net_score_vars))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 2.0
        status = solver.Solve(model)
        
        ordered_sequence, total_profit, missed_count = [], 0, 0
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for t in task_pool:
                t.calculate_metrics(solver.Value(task_starts[t.id]))
                total_profit += t.net_score
                if t.is_missed: missed_count += 1
                ordered_sequence.append(t)
            ordered_sequence.sort(key=lambda x: x.start_day)
            return ordered_sequence, total_profit, missed_count
        return SchedulixEngine.run_greedy_heap_scheduler(tasks_input, w_scale, 0)


# =====================================================================
# 3. EXPORT IO CONTROLLERS
# =====================================================================
def generate_csv_string(scheduled_tasks):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Task ID", "Name", "Department", "Start", "End", "Due Day", "Lateness", "Missed Flag", "Net Yield"])
    for t in scheduled_tasks:
        writer.writerow([t.id, t.name, t.skill, t.start_day, t.finish_day, t.due_day, t.lateness, "YES" if t.is_missed else "NO", t.net_score])
    return output.getvalue()

def generate_text_report(scheduled_tasks, total_profit, missed_count, mode_name):
    report = [
        "=" * 80, f"SCHEDULIX ENGINE RUN REPORT ({mode_name.upper()})", "=" * 80,
        f"Timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Net Profit Yield : ${total_profit:,}", f"Deadline Breaches : {missed_count}", "-" * 80,
        f"{'ID':<6}{'Task Block Name':<32}{'Start':<8}{'End':<8}{'Due':<6}{'Late':<6}{'Yield':<6}", "-" * 80
    ]
    for t in scheduled_tasks:
        report.append(f"{t.id:<6}{t.name:<32}{t.start_day:<8}{t.finish_day:<8}{t.due_day:<6}{t.lateness:<6}${t.net_score:<6}")
    return "\n".join(report)


# =====================================================================
# 4. PRIMARY SYSTEM INTERFACE WRAPPER
# =====================================================================
def run_streamlit_dashboard():
    # Seed default data into session memory safely
    if "tasks_dataset" not in st.session_state:
        st.session_state.tasks_dataset = pd.DataFrame([
            {"ID": "T1", "Task Name": "Data Architecture Ingest", "Duration": 2, "Resource Required": "Data Eng", "Due Day": 3, "Target Profit": 150, "Dependency": "None"},
            {"ID": "T2", "Task Name": "Constraint Parameter Mapping", "Duration": 2, "Resource Required": "Analytics", "Due Day": 4, "Target Profit": 120, "Dependency": "T1"},
            {"ID": "T3", "Task Name": "Solver Core Implementation", "Duration": 4, "Resource Required": "DevOps", "Due Day": 8, "Target Profit": 300, "Dependency": "T2"},
            {"ID": "T4", "Task Name": "Web Interface Assembly", "Duration": 3, "Resource Required": "UI/UX", "Due Day": 10, "Target Profit": 200, "Dependency": "T2"},
            {"ID": "T5", "Task Name": "Integration Stress Testing", "Duration": 3, "Resource Required": "QA Engine", "Due Day": 12, "Target Profit": 180, "Dependency": "T3"},
            {"ID": "T6", "Task Name": "Secure Production Rollout", "Duration": 2, "Resource Required": "SysOps", "Due Day": 14, "Target Profit": 250, "Dependency": "T5"}
        ])

    # Inject Premium Theme Overrides
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        .stApp { background-color: #060913 !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
        .block-container { max-width: 1360px; padding: 2.5rem 3.5rem !important; }
        
        [data-testid="stSidebar"] { background-color: #03050A !important; border-right: 1px solid #1E293B !important; }
        
        div[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child { display: none !important; }
        div[data-testid="stRadio"] [role="radiogroup"] { gap: 12px !important; }
        div[data-testid="stRadio"] [role="radiogroup"] > label {
            background: #0E1321 !important; border: 1px solid #1E293B !important;
            padding: 14px 18px !important; border-radius: 10px !important; cursor: pointer !important; width: 100% !important;
        }
        div[data-testid="stRadio"] [role="radiogroup"] [data-checked="true"] {
            border-color: #6366F1 !important;
            background: linear-gradient(135deg, #1E1B4B 0%, #2D2A75 100%) !important;
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3) !important;
        }
        div[data-testid="stRadio"] [role="radiogroup"] [data-checked="true"] p { color: #FFFFFF !important; font-weight: 600 !important; }

        [data-testid="stMetric"] {
            background: #0E1424 !important; border: 1px solid #232E46 !important;
            border-radius: 12px !important; padding: 22px 26px !important; position: relative;
        }
        [data-testid="stMetric"]::before {
            content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #6366F1, #06B6D4); border-top-left-radius: 12px; border-top-right-radius: 12px;
        }
        [data-testid="stMetricLabel"] p { color: #94A3B8 !important; font-size: 0.75rem !important; font-weight: 700 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
        [data-testid="stMetricValue"] div { color: #FFFFFF !important; font-size: 2.3rem !important; font-weight: 800 !important; letter-spacing: normal !important; margin-top: 6px; }
        
        div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid #232E46 !important; background: #0B101D !important; border-radius: 14px !important; padding: 28px !important; }

        h1, h2, h3 { color: #FFFFFF !important; letter-spacing: normal !important; }
        h1 { font-size: 2.65rem !important; font-weight: 800 !important; }
        h2 { font-size: 1.55rem !important; font-weight: 700 !important; margin-top: 2rem; }
        
        .premium-description { color: #CBD5E1 !important; font-size: 0.98rem !important; line-height: 1.6 !important; margin-bottom: 2rem !important; }
        .premium-body { color: #94A3B8 !important; font-size: 0.95rem !important; }

        /* Premium Extraction Form Action Button Override */
        div.stButton > button {
            background: #FFFFFF !important; color: #060913 !important; border: 1px solid #FFFFFF !important;
            border-radius: 8px !important; padding: 12px 24px !important; font-weight: 700 !important; transition: all 0.2s ease !important; width: 100%;
        }
        div.stButton > button:hover { background: transparent !important; color: #FFFFFF !important; box-shadow: 0 0 15px rgba(255, 255, 255, 0.25) !important; }
        .stDataFrame { border: 1px solid #232E46 !important; border-radius: 12px !important; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation Layout Config
    st.sidebar.markdown("<h2 style='margin-top:0px; font-size:1.35rem; color:#FFF; border:none; padding:0;'>Schedulix Core</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='color:#475569; font-size:0.7rem; font-weight:800; letter-spacing:0.1em; margin-bottom:12px;'>CONTROL CONSOLES</p>", unsafe_allow_html=True)

    active_view = st.sidebar.radio(
        label="Console View Navigation Matrix",
        options=["⚙️  Dynamic Asset Configurator", "📊  Operations Command Center", "🚀  Optimization Impact Engine"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("<br><hr style='border-color:#1E293B;'>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='color:#475569; font-size:0.7rem; font-weight:800; letter-spacing:0.1em; margin-bottom:12px;'>CRITICAL RISK SIMULATION</p>", unsafe_allow_html=True)
    w_scale = st.sidebar.slider("Workload Scale Multiplier (%)", 50, 200, 100, step=10)
    r_drop = st.sidebar.slider("Resource Constraint Deficit (%)", 0, 50, 0, step=5)

    # Transform Live State Frame into Data Class Instances
    live_tasks = []
    for _, row in st.session_state.tasks_dataset.iterrows():
        if pd.notna(row["ID"]) and str(row["ID"]).strip() != "":
            live_tasks.append(Task(
                task_id=row["ID"],
                name=row["Task Name"],
                duration=row["Duration"],
                skill=row["Resource Required"],
                due_day=row["Due Day"],
                profit=row["Target Profit"],
                dep=row["Dependency"]
            ))

    # Evaluate Engines Instantly
    h_tasks, h_profit, h_missed = SchedulixEngine.run_greedy_heap_scheduler(live_tasks, w_scale, r_drop)
    opt_tasks, opt_profit, opt_missed = SchedulixEngine.run_cpsat_solver(live_tasks, w_scale)

    # -----------------------------------------------------------------
    # ROUTE A: ASSET EDITOR
    # -----------------------------------------------------------------
    if active_view == "⚙️  Dynamic Asset Configurator":
        st.markdown("<h1>Dynamic Asset Configurator</h1>", unsafe_allow_html=True)
        st.markdown("<p class='premium-description'>Add, modify, or delete project parameters and task dependencies in real-time. Use the matrix grid bottom row to register new nodes or select and clear blocks to remove constraints.</p>", unsafe_allow_html=True)
        
        st.markdown("### Active Schedule Registry Blueprint")
        edited_df = st.data_editor(
            st.session_state.tasks_dataset,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "ID": st.column_config.TextColumn("Task ID", help="Unique identifier, e.g., T7", required=True),
                "Task Name": st.column_config.TextColumn("Task Descriptors", required=True),
                "Duration": st.column_config.NumberColumn("Base Days Duration", min_value=1, max_value=30, default=2, required=True),
                "Resource Required": st.column_config.TextColumn("Resource Variant / Skill Team", required=True),
                "Due Day": st.column_config.NumberColumn("SLA Due Benchmark Day", min_value=1, default=10, required=True),
                "Target Profit": st.column_config.NumberColumn("Gross Value Yield ($)", min_value=1, default=100, required=True),
                "Dependency": st.column_config.TextColumn("Upstream Dependency ID", help="Enter 'None' or another valid Task ID", default="None")
            }
        )
        
        if not edited_df.equals(st.session_state.tasks_dataset):
            st.session_state.tasks_dataset = edited_df.reset_index(drop=True)
            st.rerun()

        with st.expander("➕ Alternate Fast-Track Task Wizard Form"):
            with st.form("quick_add_form", clear_on_submit=True):
                f_col1, f_col2, f_col3 = st.columns(3)
                new_id = f_col1.text_input("New Task ID", placeholder="T7")
                new_name = f_col2.text_input("Task Name Descriptor", placeholder="Deployment Strategy")
                new_skill = f_col3.text_input("Resource / Team Assigned", placeholder="DevOps")
                
                f_col4, f_col5, f_col6 = st.columns(3)
                new_dur = f_col4.number_input("Duration (Days)", min_value=1, value=2)
                new_due = f_col5.number_input("SLA Deadline Day", min_value=1, value=10)
                new_profit = f_col6.number_input("Target Profit Yield ($)", min_value=10, value=200)
                
                new_dep = st.text_input("Upstream Dependency ID (Optional)", value="None")
                submit_btn = st.form_submit_button("Inject Node to Registry Matrix")
                
                if submit_btn and new_id.strip():
                    new_row = pd.DataFrame([{"ID": new_id, "Task Name": new_name, "Duration": new_dur, "Resource Required": new_skill, "Due Day": new_due, "Target Profit": new_profit, "Dependency": new_dep}])
                    st.session_state.tasks_dataset = pd.concat([st.session_state.tasks_dataset, new_row], ignore_index=True)
                    st.rerun()

    # -----------------------------------------------------------------
    # ROUTE B: MONITOR CONSOLE
    # -----------------------------------------------------------------
    elif active_view == "📊  Operations Command Center":
        st.markdown("<h1>Operations Command Center</h1>", unsafe_allow_html=True)
        st.markdown("<p class='premium-description'>High-throughput orchestration canvas rendering active graph node intervals resolved through linear constraint networks.</p>", unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Orchestrated Elements", f"{len(live_tasks)} Graph Nodes")
        m_col2.metric("Heuristic Heap Revenue", f"${h_profit:,}", delta=f"{h_missed} SLA Breaches", delta_color="inverse")
        m_col3.metric("CP-SAT Bounded Yield", f"${opt_profit:,}", delta=f"+${opt_profit - h_profit:,} Optimal Gain")

        st.markdown("<h2>Platform Execution Gantt Horizon</h2>", unsafe_allow_html=True)
        if len(opt_tasks) > 0:
            BASE_DATE = datetime.date(2026, 6, 1)
            plot_records = [{
                "Platform Execution Horizon": f" {t.id} — {t.name} ",
                "Start": BASE_DATE + datetime.timedelta(days=int(t.start_day)),
                "Finish": BASE_DATE + datetime.timedelta(days=int(t.finish_day)),
                "Resource Allocation Pool": t.skill
            } for t in opt_tasks]
            
            fig = px.timeline(
                pd.DataFrame(plot_records), x_start="Start", x_end="Finish", y="Platform Execution Horizon", 
                color="Resource Allocation Pool",
                color_discrete_sequence=["#3B82F6", "#6366F1", "#8B5CF6", "#A78BFA", "#C084FC", "#E879F9"]
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0A0F1D", font_color="#F8FAFC", font_family="Plus Jakarta Sans", height=340,
                margin=dict(l=10, r=10, t=10, b=10), showlegend=True, legend_title_text="",
                xaxis=dict(gridcolor="#1E293B", zeroline=False, linecolor="#1E293B", tickfont=dict(color="#94A3B8")),
                yaxis=dict(gridcolor="#1E293B", title_text="", linecolor="#1E293B", tickfont=dict(color="#F8FAFC"))
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.warning("Empty schedule tree layout. Use the Configurator to add operational parameters.")

        st.markdown("<h2>Data Extractions & Ledgers</h2>", unsafe_allow_html=True)
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            with st.container():
                st.markdown("### Export Runtime Tabular Matrix")
                st.markdown("<p class='premium-body'>Generates clear structural telemetry logs optimized for instant deployment data lake pipelines.</p>", unsafe_allow_html=True)
                st.download_button("Download CSV Ledger", data=generate_csv_string(opt_tasks), file_name="schedulix_matrix.csv", mime="text/csv")
        with dl_col2:
            with st.container():
                st.markdown("### Compile Verification Trace")
                st.markdown("<p class='premium-body'>Generates human-readable compliance matrices detailing engine constraint metrics for infrastructure audit verification.</p>", unsafe_allow_html=True)
                st.download_button("Download Audit Report", data=generate_text_report(opt_tasks, opt_profit, opt_missed, "Optimal Operations Frame"), file_name="schedulix_audit.txt", mime="text/plain")

    # -----------------------------------------------------------------
    # ROUTE C: DIAGNOSTICS ENGINE
    # -----------------------------------------------------------------
    elif active_view == "🚀  Optimization Impact Engine":
        st.markdown("<h1>Optimization Impact Engine</h1>", unsafe_allow_html=True)
        st.markdown("<p class='premium-description'>Side-by-side behavioral diagnostics profiling greedy local heuristics against unified global constraint matrix networks.</p>", unsafe_allow_html=True)
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            with st.container():
                st.markdown("<h3 style='border-left: 4px solid #6366F1; padding-left:12px;'>1. Local Horizon Greedy Priority Queue</h3>", unsafe_allow_html=True)
                st.markdown("<p class='premium-body' style='margin-top:10px;'>Traces sequences node-by-node utilizing processing degrees via localized Earliest Due Date (EDD) properties. This minimizes processing computational overhead but ignores critical systemic pipeline bottlenecks.</p>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Greedy Route Net Revenue Yield", f"${h_profit:,}")
                st.metric("Greedy Bounded Metric Anomalies", f"{h_missed} Nodes")

        with c_col2:
            with st.container():
                st.markdown("<h3 style='border-left: 4px solid #06B6D4; padding-left:12px;'>2. Integrated Linear CP-SAT Engine</h3>", unsafe_allow_html=True)
                st.markdown("<p class='premium-body' style='margin-top:10px;'>Maps operational constraints onto a global multi-dimensional boundary framework. By computing complex variables concurrently, it dynamically redirects sequence structures to maximize resource efficiency.</p>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("CP-SAT Bounded Yield Strategy", f"${opt_profit:,}")
                st.metric("CP-SAT Bounded Metric Anomalies", f"{opt_missed} Nodes")


if __name__ == "__main__":
    run_streamlit_dashboard()