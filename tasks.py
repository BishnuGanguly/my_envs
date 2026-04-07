from models import JobNode,  ResourceState, TaskState

from typing import Any, List, Optional, Tuple 

# _JOB_BLUEPRINTS: List[dict] = [
#     {"id": "ingest_sales",     "duration": 20, "cpu_required": 2,
#      "depends_on": [],                           "sla_deadline": None, "critical_path": False},
#     {"id": "ingest_events",    "duration": 15, "cpu_required": 1,
#      "depends_on": [],                           "sla_deadline": None, "critical_path": False},
#     {"id": "ingest_inventory", "duration": 10, "cpu_required": 1,
#      "depends_on": [],                           "sla_deadline": None, "critical_path": False},
#     {"id": "clean_sales",      "duration": 15, "cpu_required": 2,
#      "depends_on": ["ingest_sales"],             "sla_deadline": 90,   "critical_path": True},
#     {"id": "clean_events",     "duration": 10, "cpu_required": 1,
#      "depends_on": ["ingest_events"],            "sla_deadline": None, "critical_path": False},
#     {"id": "compute_revenue",  "duration": 20, "cpu_required": 3,
#      "depends_on": ["clean_sales", "clean_events"], "sla_deadline": 110, "critical_path": True},
#     {"id": "compute_metrics",  "duration": 15, "cpu_required": 2,
#      "depends_on": ["clean_events", "ingest_inventory"], "sla_deadline": None, "critical_path": False},
#     {"id": "daily_summary",    "duration": 10, "cpu_required": 1,
#      "depends_on": ["compute_revenue", "compute_metrics"], "sla_deadline": 120, "critical_path": True},
# ]
# _WORKERS_TOTAL = 3
# _CPU_TOTAL     = 8
# _TIME_BUDGET   = 180   # total episode minutes
# _STEP_SIZE     = 5     # minutes advanced by WAIT
# _TASK_DEADLINE = 120   # final SLA (minutes from episode start)

from typing import List

# -------------------------
# 1) EASY
# -------------------------
_JOB_EASY: List[dict] = [
     {"WORKERS_TOTAL": 2, "CPU_TOTAL": 4, "TIME_BUDGET": 60, "STEP_SIZE": 5, "TASK_DEADLINE": 60},
    {"id": "ingest_customers", "duration": 10, "cpu_required": 1,
     "depends_on": [], "sla_deadline": None, "critical_path": False},
    {"id": "clean_customers", "duration": 10, "cpu_required": 1,
     "depends_on": ["ingest_customers"], "sla_deadline": 40, "critical_path": True},
    {"id": "build_report", "duration": 10, "cpu_required": 1,
     "depends_on": ["clean_customers"], "sla_deadline": 60, "critical_path": True},
     
]


# -------------------------
# 2) MEDIUM
# -------------------------
_JOB_BLUEPRINTS_MEDIUM: List[dict] = [
    {"WORKERS_TOTAL": 3, "CPU_TOTAL": 6, "TIME_BUDGET": 120, "STEP_SIZE": 5, "TASK_DEADLINE": 110},
    {"id": "ingest_sales", "duration": 15, "cpu_required": 2,
     "depends_on": [], "sla_deadline": None, "critical_path": False},
    {"id": "ingest_events", "duration": 12, "cpu_required": 1,
     "depends_on": [], "sla_deadline": None, "critical_path": False},
    {"id": "clean_sales", "duration": 15, "cpu_required": 2,
     "depends_on": ["ingest_sales"], "sla_deadline": 50, "critical_path": True},
    {"id": "clean_events", "duration": 10, "cpu_required": 1,
     "depends_on": ["ingest_events"], "sla_deadline": None, "critical_path": False},
    {"id": "compute_revenue", "duration": 20, "cpu_required": 3,
     "depends_on": ["clean_sales", "clean_events"], "sla_deadline": 90, "critical_path": True},
    {"id": "summary_report", "duration": 10, "cpu_required": 1,
     "depends_on": ["compute_revenue"], "sla_deadline": 110, "critical_path": True},
]


# -------------------------
# 3) DIFFICULT
# -------------------------
_JOB_BLUEPRINTS_DIFFICULT: List[dict] = [
    {"WORKERS_TOTAL": 4, "CPU_TOTAL": 8, "TIME_BUDGET": 180, "STEP_SIZE": 5, "TASK_DEADLINE": 150},
    {"id": "ingest_sales", "duration": 20, "cpu_required": 2,
     "depends_on": [], "sla_deadline": None, "critical_path": False},
    {"id": "ingest_events", "duration": 15, "cpu_required": 1,
     "depends_on": [], "sla_deadline": None, "critical_path": False},
    {"id": "ingest_inventory", "duration": 12, "cpu_required": 1,
     "depends_on": [], "sla_deadline": None, "critical_path": False},
    {"id": "clean_sales", "duration": 15, "cpu_required": 2,
     "depends_on": ["ingest_sales"], "sla_deadline": 70, "critical_path": True},
    {"id": "clean_events", "duration": 10, "cpu_required": 1,
     "depends_on": ["ingest_events"], "sla_deadline": None, "critical_path": False},
    {"id": "clean_inventory", "duration": 10, "cpu_required": 1,
     "depends_on": ["ingest_inventory"], "sla_deadline": None, "critical_path": False},
    {"id": "compute_revenue", "duration": 20, "cpu_required": 3,
     "depends_on": ["clean_sales", "clean_events"], "sla_deadline": 100, "critical_path": True},
    {"id": "compute_metrics", "duration": 18, "cpu_required": 2,
     "depends_on": ["clean_events", "clean_inventory"], "sla_deadline": None, "critical_path": False},
    {"id": "anomaly_detection", "duration": 25, "cpu_required": 3,
     "depends_on": ["compute_metrics"], "sla_deadline": 130, "critical_path": True},
    {"id": "daily_summary", "duration": 12, "cpu_required": 1,
     "depends_on": ["compute_revenue", "anomaly_detection"], "sla_deadline": 150, "critical_path": True},
]


 # -- Build JobNodes -------------------------------------------------

#WORKERS_TOTAL, CPU_TOTAL, TIME_BUDGET, STEP_SIZE, TASK_DEADLINE can be overridden by env config, 
def build_taskstate(job_blueprints: List[dict])-> TaskState:
        jobs: List[JobNode] = []
        for bp in job_blueprints[1:]:
            status = "ready" if not bp["depends_on"] else "pending"
            jobs.append(JobNode(
                id              = bp["id"],
                duration        = bp["duration"],
                cpu_required    = bp["cpu_required"],
                depends_on      = list(bp["depends_on"]),
                sla_deadline    = bp["sla_deadline"],
                critical_path   = bp["critical_path"],
                status          = status,
                worker_occupied = False,
                cpu_occupied    = 0,
            ))

        # -- Build edges from depends_on ------------------------------------
        edges: List[Tuple[str, str]] = [
            (parent, job.id)
            for job in jobs
            for parent in job.depends_on
        ]
        critical_edges: List[Tuple[str, str]] = [
            (p, c) for (p, c) in edges
            if any(j.id == c and j.critical_path for j in jobs)
        ]

        # -- Build ResourceState (all slots free) ---------------------------
        resources = ResourceState(
            workers_total = job_blueprints[0]["WORKERS_TOTAL"],
            cpu_total     = job_blueprints[0]["CPU_TOTAL"],
            workers_free  = job_blueprints[0]["WORKERS_TOTAL"],
            cpu_free      = job_blueprints[0]["CPU_TOTAL"],
        )

        # -- Assemble TaskState ---------------------------------------------
        task_state = TaskState(
            current_time         = 0,
            time_budget          = job_blueprints[0]["TIME_BUDGET"],
            step_size            = job_blueprints[0]["STEP_SIZE"],
            task_deadline        = job_blueprints[0]["TASK_DEADLINE"],
            resources            = resources,
            jobs                 = jobs,
            edges                = edges,
            critical_path        = critical_edges,
            last_action_feedback = "Episode started. Make your first scheduling decision.",
            done                 = False,
            reward               = 0.0,
        )
        task_state.sync_job_lists()
        return task_state

def build_mul_taskstates(job_blueprints_list: List[List[dict]]) -> List[TaskState]:
    task_states: List[TaskState] = []
    for job_blueprints in job_blueprints_list:
        task_states.append(build_taskstate(job_blueprints))
    return task_states

class create_tasks_class:
     tasks = List[TaskState]
     JOB_BLUEPRINT_LIST = [_JOB_EASY, _JOB_BLUEPRINTS_MEDIUM, _JOB_BLUEPRINTS_DIFFICULT]
     def __init__(self):
        self.tasks = build_mul_taskstates(self.JOB_BLUEPRINT_LIST)