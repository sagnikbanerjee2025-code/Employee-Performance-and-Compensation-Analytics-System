"""
Employee Performance & Salary Analytics System (OOP Edition - Full)
Features:
 - MySQL-backed CRUD for employee records
 - NumPy analytics (statistics, correlation, simple regression)
 - Matplotlib charts saved as PNG files
 - Department-wise filtering for views & analytics
 - User login system with roles (admin, analyst)
 - Admin can add/remove users
 - Monthly report generator (CSV + charts) for any month
 - Passwords hashed with SHA256
 - Clean, well-documented single-file application

Modifications:
 - Users can now input emp_id manually when adding employees.
 - Added a method to count employees in a particular department and a console option to show it.
"""
import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import getpass
import hashlib
from datetime import datetime, date
from calendar import monthrange

# --------------------------
# Configuration (change as needed)
# --------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Welcome1!",  # change to your DB password
    "database": "company_db"
}

REPORT_DIR = "reports"
CHART_DIR = "charts"

# Ensure directories exist
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


# ============================
# Utility Functions
# ============================
def hash_password(password: str) -> str:
    """
    Hash a plaintext password using SHA-256 and return hex digest.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def ensure_non_empty_string(s, name="value"):
    s = str(s).strip()
    if not s:
        raise ValueError(f"{name} cannot be empty.")
    return s


# ============================
# DatabaseManager
# ============================
class DatabaseManager:
    """
    Manages database connection and ensures required tables exist.
    Tables:
      - employees(emp_id, name, department, salary, performance_score, created_at)
      - users(user_id, username, password_hash, role, created_at)
      - employee_history(hist_id, emp_id, change_type, old_value, new_value, changed_at)
    Note: emp_id is now expected to be supplied by the user (PRIMARY KEY).
    """

    def __init__(self, host="localhost", user="root", password="Welcome1!", database="company_db"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        # Ensure DB exists and create tables
        self._ensure_database_and_tables()

    def _connect_raw(self):
        return mysql.connector.connect(host=self.host, user=self.user, password=self.password)

    def connect(self):
        return mysql.connector.connect(host=self.host, user=self.user, password=self.password, database=self.database)

    def _ensure_database_and_tables(self):
        # Connect without database to create it if needed
        conn = self._connect_raw()
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        conn.commit()
        conn.close()

        # Create tables (employees with emp_id PRIMARY KEY - user-supplied)
        conn = self.connect()
        cur = conn.cursor()

        # NOTE: If you already have an employees table with AUTO_INCREMENT, you'll need to migrate or drop it
        cur.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                emp_id INT PRIMARY KEY,
                name VARCHAR(100),
                department VARCHAR(50),
                salary FLOAT,
                performance_score FLOAT,
                created_at DATETIME
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE,
                password_hash VARCHAR(128),
                role VARCHAR(20),
                created_at DATETIME
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS employee_history (
                hist_id INT AUTO_INCREMENT PRIMARY KEY,
                emp_id INT,
                change_type VARCHAR(50),
                old_value TEXT,
                new_value TEXT,
                changed_at DATETIME,
                FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        conn.close()

        # Ensure at least one admin user exists (default admin/admin if not present)
        self._ensure_default_admin()

    def _ensure_default_admin(self):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        if count == 0:
            default_user = "admin"
            default_pass = "admin"  # encourage changing this on first run
            hashed = hash_password(default_pass)
            cur.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (%s, %s, %s, %s)",
                (default_user, hashed, "admin", datetime.now())
            )
            conn.commit()
            print("Default admin user created: username='admin', password='admin'. Please change it after login.")
        conn.close()

    # User management queries
    def add_user(self, username: str, password_plain: str, role: str):
        username = ensure_non_empty_string(username, "username")
        password_hash = hash_password(password_plain)
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (%s, %s, %s, %s)",
                    (username, password_hash, role, datetime.now()))
        conn.commit()
        conn.close()

    def remove_user(self, username: str):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE username=%s", (username,))
        conn.commit()
        conn.close()

    def authenticate_user(self, username: str, password_plain: str):
        """
        Returns user dict if authenticated, else None.
        user dict contains: user_id, username, role
        """
        conn = self.connect()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT user_id, username, password_hash, role FROM users WHERE username=%s", (username,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        if hash_password(password_plain) == row["password_hash"]:
            return {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}
        return None

    # Employee CRUD methods
    def insert_employee(self, emp_id, name, department, salary, performance_score):
        """
        Insert an employee with user-supplied emp_id.
        Returns the emp_id on success.
        Raises a ValueError if emp_id already exists or on other DB errors.
        """
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO employees (emp_id, name, department, salary, performance_score, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (emp_id, name, department, salary, performance_score, datetime.now())
            )
            conn.commit()
        except mysql.connector.Error as err:
            conn.rollback()
            # Duplicate primary key
            if err.errno in (1062,):  # duplicate entry error code (MySQL)
                raise ValueError(f"Employee ID {emp_id} already exists. Please pick a different ID.") from err
            else:
                raise
        finally:
            conn.close()
        return emp_id

    def fetch_all_employees(self, department_filter: str = None):
        conn = self.connect()
        cur = conn.cursor()
        if department_filter:
            cur.execute("SELECT * FROM employees WHERE department=%s ORDER BY emp_id", (department_filter,))
        else:
            cur.execute("SELECT * FROM employees ORDER BY emp_id")
        rows = cur.fetchall()
        conn.close()
        # rows: list of tuples (emp_id, name, department, salary, performance_score, created_at)
        return rows

    def fetch_employee_by_id(self, emp_id):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM employees WHERE emp_id=%s", (emp_id,))
        row = cur.fetchone()
        conn.close()
        return row

    def update_employee(self, emp_id, salary=None, performance_score=None, department=None, name=None):
        # Save old values to history
        old = self.fetch_employee_by_id(emp_id)
        if not old:
            raise ValueError("Employee not found.")

        conn = self.connect()
        cur = conn.cursor()

        if salary is not None:
            cur.execute("UPDATE employees SET salary=%s WHERE emp_id=%s", (salary, emp_id))
            cur.execute("INSERT INTO employee_history (emp_id, change_type, old_value, new_value, changed_at) VALUES (%s, %s, %s, %s, %s)",
                        (emp_id, "salary_update", str(old[3]), str(salary), datetime.now()))
        if performance_score is not None:
            cur.execute("UPDATE employees SET performance_score=%s WHERE emp_id=%s", (performance_score, emp_id))
            cur.execute("INSERT INTO employee_history (emp_id, change_type, old_value, new_value, changed_at) VALUES (%s, %s, %s, %s, %s)",
                        (emp_id, "performance_update", str(old[4]), str(performance_score), datetime.now()))
        if department is not None:
            cur.execute("UPDATE employees SET department=%s WHERE emp_id=%s", (department, emp_id))
            cur.execute("INSERT INTO employee_history (emp_id, change_type, old_value, new_value, changed_at) VALUES (%s, %s, %s, %s, %s)",
                        (emp_id, "department_update", str(old[2]), department, datetime.now()))
        if name is not None:
            cur.execute("UPDATE employees SET name=%s WHERE emp_id=%s", (name, emp_id))
            cur.execute("INSERT INTO employee_history (emp_id, change_type, old_value, new_value, changed_at) VALUES (%s, %s, %s, %s, %s)",
                        (emp_id, "name_update", str(old[1]), name, datetime.now()))

        conn.commit()
        conn.close()

    def delete_employee(self, emp_id):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM employees WHERE emp_id=%s", (emp_id,))
        conn.commit()
        conn.close()

    # History fetch
    def fetch_history_for_employee(self, emp_id):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT hist_id, change_type, old_value, new_value, changed_at FROM employee_history WHERE emp_id=%s ORDER BY changed_at DESC", (emp_id,))
        rows = cur.fetchall()
        conn.close()
        return rows

    # Fetch employees created in a given month
    def fetch_employees_for_month(self, year: int, month: int, department_filter: str = None):
        start = datetime(year, month, 1)
        end_day = monthrange(year, month)[1]
        end = datetime(year, month, end_day, 23, 59, 59)
        conn = self.connect()
        cur = conn.cursor()
        if department_filter:
            cur.execute("SELECT * FROM employees WHERE created_at BETWEEN %s AND %s AND department=%s ORDER BY emp_id", (start, end, department_filter))
        else:
            cur.execute("SELECT * FROM employees WHERE created_at BETWEEN %s AND %s ORDER BY emp_id", (start, end))
        rows = cur.fetchall()
        conn.close()
        return rows


# ============================
# Employee Entity Class
# ============================
class Employee:
    """
    Simple container for employee data.
    """

    def __init__(self, name, department, salary, performance_score, emp_id=None, created_at=None):
        self.emp_id = emp_id
        self.name = name
        self.department = department
        self.salary = float(salary)
        self.performance_score = float(performance_score)
        self.created_at = created_at

    @classmethod
    def from_db_row(cls, row):
        if not row:
            return None
        # row order: emp_id, name, department, salary, performance_score, created_at
        emp_id, name, department, salary, perf, created_at = row
        return cls(name=name, department=department, salary=salary, performance_score=perf, emp_id=emp_id, created_at=created_at)

    def __repr__(self):
        return f"Employee(emp_id={self.emp_id}, name='{self.name}', department='{self.department}', salary={self.salary:.2f}, perf={self.performance_score:.2f})"


# ============================
# EmployeeManager (Business Logic)
# ============================
class EmployeeManager:
    """
    High-level employee management using DatabaseManager.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    # Now expects emp_id provided by caller
    def add_employee(self, emp_id, name, department, salary, performance_score):
        name = ensure_non_empty_string(name, "Name")
        department = ensure_non_empty_string(department, "Department")
        salary = safe_float(salary)
        performance_score = safe_float(performance_score)
        # Insert and return emp_id (or raise ValueError on duplicate)
        new_id = self.db.insert_employee(emp_id, name, department, salary, performance_score)
        return new_id

    def list_employees(self, department_filter: str = None):
        rows = self.db.fetch_all_employees(department_filter)
        return [Employee.from_db_row(r) for r in rows]

    def get_employee(self, emp_id):
        row = self.db.fetch_employee_by_id(emp_id)
        return Employee.from_db_row(row)

    def update_employee(self, emp_id, salary=None, performance_score=None, department=None, name=None):
        # Validate before calling DB
        if salary is not None:
            salary = safe_float(salary)
        if performance_score is not None:
            performance_score = safe_float(performance_score)
        if department is not None:
            department = ensure_non_empty_string(department, "Department")
        if name is not None:
            name = ensure_non_empty_string(name, "Name")
        self.db.update_employee(emp_id, salary, performance_score, department, name)

    def delete_employee(self, emp_id):
        # Basic check
        if not self.get_employee(emp_id):
            raise ValueError("Employee not found.")
        self.db.delete_employee(emp_id)

    def get_history(self, emp_id):
        return self.db.fetch_history_for_employee(emp_id)

    # NEW: Count employees in a department
    def count_by_department(self, department):
        department = ensure_non_empty_string(department, "Department")
        rows = self.db.fetch_all_employees(department)
        return len(rows)


# ============================
# AnalyticsEngine (NumPy + Matplotlib)
# ============================
class AnalyticsEngine:
    """
    Uses NumPy for statistics and Matplotlib for charts.
    Accepts an EmployeeManager instance to fetch data.
    """

    def __init__(self, emp_manager: EmployeeManager):
        self.emp_manager = emp_manager

    def _get_numpy_data(self, department_filter: str = None):
        employees = self.emp_manager.list_employees(department_filter)
        if not employees:
            return np.array([])  # empty
        arr = np.array([[e.salary, e.performance_score] for e in employees], dtype=float)
        return arr

    # Basic numeric stats
    def salary_statistics(self, department_filter: str = None):
        arr = self._get_numpy_data(department_filter)
        if arr.size == 0:
            print("No data available for salary statistics.")
            return None
        salaries = arr[:, 0]
        stats = {
            "mean": float(np.mean(salaries)),
            "median": float(np.median(salaries)),
            "max": float(np.max(salaries)),
            "min": float(np.min(salaries)),
            "std": float(np.std(salaries)),
            "count": int(salaries.size)
        }
        return stats

    def performance_statistics(self, department_filter: str = None):
        arr = self._get_numpy_data(department_filter)
        if arr.size == 0:
            print("No data available for performance statistics.")
            return None
        perf = arr[:, 1]
        stats = {
            "mean": float(np.mean(perf)),
            "max": float(np.max(perf)),
            "min": float(np.min(perf)),
            "median": float(np.median(perf)),
            "std": float(np.std(perf)),
            "count": int(perf.size)
        }
        return stats

    # Correlation between salary and performance
    def correlation(self, department_filter: str = None):
        arr = self._get_numpy_data(department_filter)
        if arr.size == 0:
            print("No data available for correlation.")
            return None
        salaries = arr[:, 0]
        perf = arr[:, 1]
        if salaries.size < 2:
            print("Not enough data to compute correlation.")
            return None
        corr = float(np.corrcoef(salaries, perf)[0, 1])
        return corr

    # Simple linear regression (salary = m * perf + c) using least squares
    def fit_salary_from_performance(self, department_filter: str = None):
        arr = self._get_numpy_data(department_filter)
        if arr.size == 0 or arr.shape[0] < 2:
            print("Not enough data to fit regression.")
            return None
        salaries = arr[:, 0]
        perf = arr[:, 1]
        A = np.vstack([perf, np.ones(len(perf))]).T
        m, c = np.linalg.lstsq(A, salaries, rcond=None)[0]
        return float(m), float(c)

    def predict_salary(self, perf_score: float, department_filter: str = None):
        model = self.fit_salary_from_performance(department_filter)
        if not model:
            return None
        m, c = model
        return m * perf_score + c

    # Charts using matplotlib (saved to disk). Each returns filepath.
    def plot_salary_distribution(self, department_filter: str = None):
        arr = self._get_numpy_data(department_filter)
        if arr.size == 0:
            print("No data to plot salary distribution.")
            return None
        salaries = arr[:, 0]
        fig, ax = plt.subplots()
        ax.hist(salaries, bins=10)
        title = "Salary Distribution"
        if department_filter:
            title += f" - {department_filter}"
        ax.set_title(title)
        ax.set_xlabel("Salary")
        ax.set_ylabel("Number of employees")
        filename = os.path.join(CHART_DIR, f"salary_distribution_{department_filter or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filename)
        plt.close(fig)
        return filename

    def plot_performance_distribution(self, department_filter: str = None):
        arr = self._get_numpy_data(department_filter)
        if arr.size == 0:
            print("No data to plot performance distribution.")
            return None
        perf = arr[:, 1]
        fig, ax = plt.subplots()
        ax.hist(perf, bins=10)
        title = "Performance Score Distribution"
        if department_filter:
            title += f" - {department_filter}"
        ax.set_title(title)
        ax.set_xlabel("Performance Score")
        ax.set_ylabel("Number of employees")
        filename = os.path.join(CHART_DIR, f"performance_distribution_{department_filter or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filename)
        plt.close(fig)
        return filename

    def plot_salary_vs_performance(self, department_filter: str = None):
        arr = self._get_numpy_data(department_filter)
        if arr.size == 0:
            print("No data to plot salary vs performance.")
            return None
        salaries = arr[:, 0]
        perf = arr[:, 1]
        fig, ax = plt.subplots()
        ax.scatter(perf, salaries)
        title = "Salary vs Performance"
        if department_filter:
            title += f" - {department_filter}"
        ax.set_title(title)
        ax.set_xlabel("Performance Score")
        ax.set_ylabel("Salary")
        # Add regression line if possible
        if len(perf) >= 2:
            m, c = np.linalg.lstsq(np.vstack([perf, np.ones(len(perf))]).T, salaries, rcond=None)[0]
            xs = np.linspace(min(perf), max(perf), 100)
            ys = m * xs + c
            ax.plot(xs, ys, linestyle='--')
        filename = os.path.join(CHART_DIR, f"salary_vs_perf_{department_filter or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filename)
        plt.close(fig)
        return filename

    # --------------------
    # New dashboard methods
    # --------------------

    def department_comparison_dashboard(self):
        """
        Compare average salary and average performance across all departments.
        Saves a bar chart and prints department-wise averages.
        """
        employees = self.emp_manager.list_employees()
        if not employees:
            print("No employee data available.")
            return None

        dept_map = {}
        for e in employees:
            dept_map.setdefault(e.department, []).append(e)

        dept_names = []
        avg_salaries = []
        avg_performances = []

        for dept, emps in dept_map.items():
            salaries = [emp.salary for emp in emps]
            perfs = [emp.performance_score for emp in emps]
            dept_names.append(dept)
            avg_salaries.append(float(np.mean(salaries)))
            avg_performances.append(float(np.mean(perfs)))

        # Print summary to console
        print("\n--- Department Comparison: Avg Salary vs Avg Performance ---")
        for i, d in enumerate(dept_names):
            print(f"{d}: Avg Salary = {avg_salaries[i]:.2f}, Avg Performance = {avg_performances[i]:.2f}")

        # Plot: side-by-side bars
        x = np.arange(len(dept_names))
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - width/2, avg_salaries, width, label='Avg Salary')
        ax.bar(x + width/2, avg_performances, width, label='Avg Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(dept_names, rotation=45, ha='right')
        ax.set_title("Department Comparison (Avg Salary vs Avg Performance)")
        ax.legend()
        plt.tight_layout()

        filename = os.path.join(CHART_DIR, f"department_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filename)
        plt.close(fig)
        print(f"Department comparison chart saved to: {filename}")
        return filename

    def employee_comparison_dashboard(self, emp_id: int):
        """
        For a given employee ID, compare that employee's salary & performance to
        their department average and overall company average.
        Saves a small comparison chart and prints the numbers.
        """
        emp = self.emp_manager.get_employee(emp_id)
        if not emp:
            print("Employee not found.")
            return None

        # gather department and company arrays
        dept_emps = self.emp_manager.list_employees(emp.department)
        all_emps = self.emp_manager.list_employees()

        if not dept_emps or not all_emps:
            print("Insufficient data for comparison (need at least one dept and company data).")
            return None

        dept_salaries = np.array([e.salary for e in dept_emps], dtype=float)
        dept_perfs = np.array([e.performance_score for e in dept_emps], dtype=float)
        all_salaries = np.array([e.salary for e in all_emps], dtype=float)
        all_perfs = np.array([e.performance_score for e in all_emps], dtype=float)

        dept_avg_salary = float(np.mean(dept_salaries))
        dept_avg_perf = float(np.mean(dept_perfs))
        all_avg_salary = float(np.mean(all_salaries))
        all_avg_perf = float(np.mean(all_perfs))

        # Print comparison
        print(f"\n--- Employee Comparison for {emp.name} (ID {emp.emp_id}) ---")
        print(f"Department: {emp.department}")
        print(f"Employee Salary: {emp.salary:.2f} | Employee Performance: {emp.performance_score:.2f}")
        print(f"Department Avg Salary: {dept_avg_salary:.2f} | Department Avg Perf: {dept_avg_perf:.2f}")
        print(f"Overall Avg Salary: {all_avg_salary:.2f} | Overall Avg Perf: {all_avg_perf:.2f}")
        # --- Percent Difference Calculations ---
        dept_salary_diff = ((emp.salary - dept_avg_salary) / dept_avg_salary) * 100
        dept_perf_diff = ((emp.performance_score - dept_avg_perf) / dept_avg_perf) * 100
        overall_salary_diff = ((emp.salary - all_avg_salary) / all_avg_salary) * 100
        overall_perf_diff = ((emp.performance_score - all_avg_perf) / all_avg_perf) * 100

        print("\n--- Percent Difference ---")
        print(f"Salary vs Dept Avg: {dept_salary_diff:+.2f}%")
        print(f"Performance vs Dept Avg: {dept_perf_diff:+.2f}%")
        print(f"Salary vs Overall Avg: {overall_salary_diff:+.2f}%")
        print(f"Performance vs Overall Avg: {overall_perf_diff:+.2f}%")


        # Build chart: Employee vs Dept Avg vs Overall Avg (two grouped bars)
        labels = ["Employee", "Dept Avg", "Overall Avg"]
        salaries = [emp.salary, dept_avg_salary, all_avg_salary]
        performances = [emp.performance_score, dept_avg_perf, all_avg_perf]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - width/2, salaries, width, label='Salary')
        ax.bar(x + width/2, performances, width, label='Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{emp.name} - Salary & Performance Comparison")
        ax.legend()
        plt.tight_layout()

        filename = os.path.join(CHART_DIR, f"employee_comparison_{emp.emp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filename)
        plt.close(fig)
        print(f"Employee comparison chart saved to: {filename}")
        return filename


# ============================
# ReportGenerator
# ============================
class ReportGenerator:
    """
    Produces CSV reports and associated charts for a given month (or overall).
    """

    def __init__(self, emp_manager: EmployeeManager, analytics: AnalyticsEngine, db_manager: DatabaseManager):
        self.emp_manager = emp_manager
        self.analytics = analytics
        self.db_manager = db_manager

    def export_all_employees_csv(self, filename: str = None, department_filter: str = None):
        employees = self.emp_manager.list_employees(department_filter)
        if not filename:
            filename = os.path.join(REPORT_DIR, f"employees_{department_filter or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(filename, mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Department", "Salary", "Performance", "Created At"])
            for e in employees:
                writer.writerow([e.emp_id, e.name, e.department, f"{e.salary:.2f}", f"{e.performance_score:.2f}", e.created_at])
        return filename

    def generate_monthly_report(self, year: int, month: int, department_filter: str = None):
        """
        Generates a CSV and charts for the given month. The CSV includes employees created in that month.
        Also produces summary statistics.
        """
        rows = self.db_manager.fetch_employees_for_month(year, month, department_filter)
        # rows have same schema as employees table
        if not rows:
            print("No employees created in that month for the chosen filter.")
            return None

        # CSV for the month
        filename_csv = os.path.join(REPORT_DIR, f"monthly_report_{year}_{month:02d}_{department_filter or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(filename_csv, mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Department", "Salary", "Performance", "Created At"])
            for r in rows:
                writer.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", r[5]])

        # Simple stats for that month (based on the rows)
        arr = np.array([[r[3], r[4]] for r in rows], dtype=float)
        salaries = arr[:, 0]
        perf = arr[:, 1]
        stats = {
            "salary_mean": float(np.mean(salaries)),
            "salary_median": float(np.median(salaries)),
            "salary_min": float(np.min(salaries)),
            "salary_max": float(np.max(salaries)),
            "salary_std": float(np.std(salaries)),
            "perf_mean": float(np.mean(perf)),
            "perf_median": float(np.median(perf)),
            "perf_min": float(np.min(perf)),
            "perf_max": float(np.max(perf)),
            "perf_std": float(np.std(perf)),
            "count": int(arr.shape[0])
        }

        # Produce charts limited to this month's rows
        # Salary histogram
        fig1, ax1 = plt.subplots()
        ax1.hist(salaries, bins=8)
        ax1.set_title(f"Salary Distribution - {year}-{month:02d} - {department_filter or 'All'}")
        ax1.set_xlabel("Salary")
        ax1.set_ylabel("Count")
        chart_salary = os.path.join(CHART_DIR, f"monthly_salary_{year}_{month:02d}_{department_filter or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig1.savefig(chart_salary)
        plt.close(fig1)

        # Performance histogram
        fig2, ax2 = plt.subplots()
        ax2.hist(perf, bins=8)
        ax2.set_title(f"Performance Distribution - {year}-{month:02d} - {department_filter or 'All'}")
        ax2.set_xlabel("Performance Score")
        ax2.set_ylabel("Count")
        chart_perf = os.path.join(CHART_DIR, f"monthly_perf_{year}_{month:02d}_{department_filter or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig2.savefig(chart_perf)
        plt.close(fig2)

        # Salary vs Performance scatter + regression line
        fig3, ax3 = plt.subplots()
        ax3.scatter(perf, salaries)
        if len(perf) >= 2:
            m, c = np.linalg.lstsq(np.vstack([perf, np.ones(len(perf))]).T, salaries, rcond=None)[0]
            xs = np.linspace(min(perf), max(perf), 100)
            ys = m * xs + c
            ax3.plot(xs, ys, linestyle='--')
        ax3.set_title(f"Salary vs Performance - {year}-{month:02d} - {department_filter or 'All'}")
        ax3.set_xlabel("Performance Score")
        ax3.set_ylabel("Salary")
        chart_scatter = os.path.join(CHART_DIR, f"monthly_scatter_{year}_{month:02d}_{department_filter or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig3.savefig(chart_scatter)
        plt.close(fig3)

        # Summarize stats into a small CSV
        summary_csv = os.path.join(REPORT_DIR, f"monthly_summary_{year}_{month:02d}_{department_filter or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(summary_csv, mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in stats.items():
                writer.writerow([k, v])

        return {
            "data_csv": filename_csv,
            "summary_csv": summary_csv,
            "charts": {
                "salary_hist": chart_salary,
                "performance_hist": chart_perf,
                "scatter": chart_scatter
            },
            "stats": stats
        }


# ============================
# UserManager (for login & admin)
# ============================
class UserManager:
    """
    Provides higher-level user management functions built on DatabaseManager.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def login(self):
        username = input("Username: ").strip()
        password = getpass.getpass("Password: ").strip()
        user = self.db.authenticate_user(username, password)
        if not user:
            print("Authentication failed. Please check your username and password.")
            return None
        print(f"Login successful. Role: {user['role']}")
        return user

    def add_user(self, username: str, password_plain: str, role: str):
        role = role.strip().lower()
        if role not in ("admin", "analyst"):
            raise ValueError("Role must be 'admin' or 'analyst'.")
        self.db.add_user(username, password_plain, role)
        print(f"User '{username}' added with role '{role}'.")

    def remove_user(self, username: str):
        self.db.remove_user(username)
        print(f"User '{username}' removed (if existed).")


# ============================
# Console Application
# ============================
class ConsoleApp:
    """
    Main interactive console application handling user sessions and commands.
    """

    def __init__(self):
        self.db = DatabaseManager(host=DB_CONFIG["host"], user=DB_CONFIG["user"], password=DB_CONFIG["password"], database=DB_CONFIG["database"])
        self.user_mgr = UserManager(self.db)
        self.emp_mgr = EmployeeManager(self.db)
        self.analytics = AnalyticsEngine(self.emp_mgr)
        self.reporter = ReportGenerator(self.emp_mgr, self.analytics, self.db)
        self.current_user = None

    # ---------- Authentication ----------
    def login_flow(self):
        print("Please log in.")
        user = self.user_mgr.login()
        if not user:
            return False
        self.current_user = user
        return True

    def require_role(self, allowed_roles):
        """
        Decorator-like check for role-based access; used in the interactive flow.
        """
        if self.current_user is None:
            raise PermissionError("No user logged in.")
        if self.current_user["role"] not in allowed_roles:
            raise PermissionError("You do not have permission to perform this action.")

    # ---------- Interactive Menus ----------
    def run(self):
        # Login loop
        logged_in = False
        while not logged_in:
            try:
                logged_in = self.login_flow()
            except Exception as e:
                print(f"Login error: {e}")
                return

        # Main loop
        while True:
            try:
                self._print_main_menu()
                choice = input("Enter option: ").strip()
                if choice == "1":
                    self._add_employee_flow()
                elif choice == "2":
                    self._view_employees_flow()
                elif choice == "3":
                    self._update_employee_flow()
                elif choice == "4":
                    self._delete_employee_flow()
                elif choice == "5":
                    self._analytics_menu()
                elif choice == "6":
                    self._reports_menu()
                elif choice == "7":
                    if self.current_user["role"] == "admin":
                        self._user_management_menu()
                    else:
                        print("Only admin users can access user management.")
                elif choice == "0":
                    print("Exiting. Goodbye.")
                    break
                else:
                    print("Invalid option. Please choose again.")
            except PermissionError as pe:
                print(f"Permission error: {pe}")
            except Exception as exc:
                print(f"An error occurred: {exc}")

    def _print_main_menu(self):
        print("\n===== Employee Performance & Salary Analytics System =====")
        print("1. Add Employee")
        print("2. View Employees")
        print("3. Update Employee")
        print("4. Delete Employee")
        print("5. Analytics")
        print("6. Reports")
        if self.current_user and self.current_user["role"] == "admin":
            print("7. User Management (admin only)")
        print("0. Exit")
        print("=========================================================")

    # ---------- Employee flows ----------
    def _add_employee_flow(self):
        try:
            # <-- NEW: user supplies emp_id here -->
            emp_id_str = input("Employee ID (integer): ").strip()
            emp_id = int(emp_id_str)
            name = input("Name: ").strip()
            department = input("Department: ").strip()
            salary = input("Salary: ").strip()
            perf = input("Performance score (1-10): ").strip()
            new_id = self.emp_mgr.add_employee(emp_id, name, department, salary, perf)
            print(f"Employee added with ID {new_id}.")
        except ValueError as ve:
            print(f"Failed to add employee: {ve}")
        except Exception as e:
            print(f"Failed to add employee: {e}")

    def _view_employees_flow(self):
        try:
            dept = input("Filter by department (leave blank for all): ").strip()
            dept_filter = dept if dept else None
            employees = self.emp_mgr.list_employees(dept_filter)
            if not employees:
                print("No employees found.")
                return
            print(f"Listing employees (department filter: {dept_filter or 'all'})")
            for e in employees:
                print(f"ID: {e.emp_id} | Name: {e.name} | Dept: {e.department} | Salary: {e.salary:.2f} | Perf: {e.performance_score:.2f} | Created: {e.created_at}")
            # Optionally show employee history
            show_history = input("Would you like to view history for an employee? (y/N): ").strip().lower()
            if show_history == "y":
                emp_id = int(input("Enter employee ID: ").strip())
                history = self.emp_mgr.get_history(emp_id)
                if not history:
                    print("No history found for that employee.")
                else:
                    print("History (most recent first):")
                    for h in history:
                        print(f"{h[4]} | {h[1]} | from '{h[2]}' to '{h[3]}'")

            # <-- NEW: ask if user wants to count employees in a department -->
            count_choice = input("Would you like to count employees in a department? (y/N): ").strip().lower()
            if count_choice == "y":
                dept_to_count = input("Enter department name: ").strip()
                try:
                    cnt = self.emp_mgr.count_by_department(dept_to_count)
                    print(f"Number of employees in '{dept_to_count}': {cnt}")
                except Exception as e:
                    print(f"Error counting department employees: {e}")

        except Exception as e:
            print(f"Error viewing employees: {e}")

    def _update_employee_flow(self):
        try:
            emp_id = int(input("Enter employee ID to update: ").strip())
            emp = self.emp_mgr.get_employee(emp_id)
            if not emp:
                print("Employee not found.")
                return
            print(f"Current: {emp}")
            name = input("New name (leave blank to keep current): ").strip()
            department = input("New department (leave blank to keep current): ").strip()
            salary = input("New salary (leave blank to keep current): ").strip()
            perf = input("New performance score (leave blank to keep current): ").strip()
            kwargs = {}
            if name:
                kwargs["name"] = name
            if department:
                kwargs["department"] = department
            if salary:
                kwargs["salary"] = salary
            if perf:
                kwargs["performance_score"] = perf
            if not kwargs:
                print("No changes provided.")
                return
            self.emp_mgr.update_employee(emp_id, **kwargs)
            print("Employee updated.")
        except Exception as e:
            print(f"Error updating employee: {e}")

    def _delete_employee_flow(self):
        try:
            emp_id = int(input("Enter employee ID to delete: ").strip())
            confirm = input("Are you sure you want to delete this employee? Type 'yes' to confirm: ").strip().lower()
            if confirm != "yes":
                print("Deletion cancelled.")
                return
            self.emp_mgr.delete_employee(emp_id)
            print("Employee deleted.")
        except Exception as e:
            print(f"Error deleting employee: {e}")

    # ---------- Analytics menu ----------
    def _analytics_menu(self):
        while True:
            print("\n--- Analytics Menu ---")
            print("1. Salary statistics")
            print("2. Performance statistics")
            print("3. Correlation (Salary vs Performance)")
            print("4. Fit regression model (Salary = m*Perf + c)")
            print("5. Predict salary from performance score")
            print("6. Generate and save charts (histograms, scatter)")
            print("7. Department Comparison Dashboard (Avg salary vs Avg performance)")
            print("8. Employee Comparison Dashboard (employee vs dept & overall)")
            print("0. Back to main")
            choice = input("Choose option: ").strip()
            try:
                if choice == "1":
                    dept = input("Department filter (leave blank for all): ").strip() or None
                    stats = self.analytics.salary_statistics(dept)
                    if stats:
                        print("Salary statistics:")
                        for k, v in stats.items():
                            print(f"  {k}: {v}")
                elif choice == "2":
                    dept = input("Department filter (leave blank for all): ").strip() or None
                    stats = self.analytics.performance_statistics(dept)
                    if stats:
                        print("Performance statistics:")
                        for k, v in stats.items():
                            print(f"  {k}: {v}")
                elif choice == "3":
                    dept = input("Department filter (leave blank for all): ").strip() or None
                    corr = self.analytics.correlation(dept)
                    if corr is not None:
                        print(f"Correlation (salary vs performance): {corr:.4f}")
                elif choice == "4":
                    dept = input("Department filter (leave blank for all): ").strip() or None
                    model = self.analytics.fit_salary_from_performance(dept)
                    if model:
                        m, c = model
                        print(f"Fitted model: Salary = {m:.4f} * Performance + {c:.4f}")
                elif choice == "5":
                    dept = input("Department filter (leave blank for all): ").strip() or None
                    score = float(input("Enter performance score: ").strip())
                    pred = self.analytics.predict_salary(score, dept)
                    if pred is not None:
                        print(f"Predicted salary: {pred:.2f}")
                elif choice == "6":
                    dept = input("Department filter (leave blank for all): ").strip() or None
                    sfile = self.analytics.plot_salary_distribution(dept)
                    pfile = self.analytics.plot_performance_distribution(dept)
                    spfile = self.analytics.plot_salary_vs_performance(dept)
                    print("Charts generated:")
                    for f in (sfile, pfile, spfile):
                        if f:
                            print(f"  {f}")
                elif choice == "7":
                    # Department comparison: avg salary vs avg performance across departments
                    self.analytics.department_comparison_dashboard()
                elif choice == "8":
                    # Employee comparison: employee vs dept and overall
                    try:
                        emp_id = int(input("Enter employee ID: ").strip())
                        self.analytics.employee_comparison_dashboard(emp_id)
                    except ValueError:
                        print("Invalid employee ID.")
                elif choice == "0":
                    break
                else:
                    print("Invalid option.")
            except Exception as e:
                print(f"Analytics error: {e}")

    # ---------- Reports menu ----------
    def _reports_menu(self):
        while True:
            print("\n--- Reports Menu ---")
            print("1. Export all employees CSV")
            print("2. Generate monthly report (CSV + charts)")
            print("0. Back to main")
            choice = input("Choose option: ").strip()
            try:
                if choice == "1":
                    dept = input("Department filter (leave blank for all): ").strip() or None
                    fname = self.reporter.export_all_employees_csv(department_filter=dept)
                    print(f"Exported employees CSV: {fname}")
                elif choice == "2":
                    year = int(input("Enter year (YYYY): ").strip())
                    month = int(input("Enter month (1-12): ").strip())
                    dept = input("Department filter (leave blank for all): ").strip() or None
                    result = self.reporter.generate_monthly_report(year, month, dept)
                    if result:
                        print("Monthly report generated:")
                        print(f"  Data CSV: {result['data_csv']}")
                        print(f"  Summary CSV: {result['summary_csv']}")
                        for k, v in result["charts"].items():
                            print(f"  Chart ({k}): {v}")
                        print("Stats:")
                        for k, v in result["stats"].items():
                            print(f"  {k}: {v}")
                elif choice == "0":
                    break
                else:
                    print("Invalid option.")
            except Exception as e:
                print(f"Report error: {e}")

    # ---------- User management (admin only) ----------
    def _user_management_menu(self):
        if self.current_user["role"] != "admin":
            print("Access denied. Admins only.")
            return
        while True:
            print("\n--- User Management (admin) ---")
            print("1. Add user")
            print("2. Remove user")
            print("0. Back to main")
            choice = input("Option: ").strip()
            try:
                if choice == "1":
                    username = input("New username: ").strip()
                    password = getpass.getpass("Password: ").strip()
                    role = input("Role (admin/analyst): ").strip().lower()
                    self.user_mgr.add_user(username, password, role)
                elif choice == "2":
                    username = input("Username to remove: ").strip()
                    confirm = input(f"Type 'delete {username}' to confirm removal: ").strip()
                    if confirm == f"delete {username}":
                        self.user_mgr.remove_user(username)
                    else:
                        print("Confirmation mismatch. Cancelled.")
                elif choice == "0":
                    break
                else:
                    print("Invalid option.")
            except Exception as e:
                print(f"User management error: {e}")


# ============================
# Main Runner
# ============================
def main():
    app = ConsoleApp()
    app.run()


if __name__ == "__main__":
    main()
