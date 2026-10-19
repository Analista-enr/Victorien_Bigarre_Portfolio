### **Presentation of Projects**

---

#### **PROJECT 1: Customs Scheduling Optimization**

**Context:**
This problem is inspired by INFORMS Transactions on Education and aims to **minimize the total number of customs officers** in an airport while respecting operational constraints.

**Parameters:**
- **Mandatory inputs:**
  - Schedule of arrivals (origin, date, time, number of passengers).
  - Dataset of average processing time per passenger by origin.
  - Maximum average waiting time for passengers in the queue.

- **Optional inputs:**
  - Cost of a customs officer.
  - Cost of one minute of waiting in the queue.

- **Default parameters:**
  - Work schedule: 40 hours/week with 2 consecutive days off.
  - Each worker is assigned to one quarter of the day.
  - All inputs have default definitions.

**Modeling Decisions:**
- Passenger numbers are converted into **passenger-minutes** to standardize queue metrics.
- The objective function minimizes the **total number of customs officers** and the **mean passenger-minutes in the queue** over the studied period.
- The average waiting time constraint is satisfied using the mean passenger-minutes in the queue.

**Results:**
- Optimal number of customs officers and their schedule.
- Comparison with a naive approximation.
- Dashboard visualizing queue dynamics, stress periods, and Pareto-optimal solutions based on optional inputs.

---

#### **PROJECT 2: Train Assignment Optimization at Marseille St-Charles Station**

**Context:**
After experiencing delays at Marseille St-Charles Station, I explored the complexity of train assignment management. This project simulates how to reassign trains efficiently under real-world constraints.

**Parameters:**
- **Mandatory inputs:**
  - Schedule of arrivals and departures for all trains.
  - Pre-assigned train assignments.
  - Number of available platforms.

- **Default parameters:**
  - Penalty weight (adjustable if needed).
  - Default dataset based on real SNCF data from Marseille St-Charles.

**Modeling Decisions:**
- SNCF must announce platform assignments **20 minutes before departure**.
- A **5-minute buffer** after departure is used to avoid collisions.
- Due to the slowness of linear solvers, a **metaheuristic approach** was adopted.
- Benchmarking highlighted the superiority of the **PSO + ASA algorithm**.
- The model uses a **time overlap matrix** to compute penalties for assignment collisions.

**Results:**
- Optimal assignment for each train.
- Total overlap metric to evaluate solution quality.
- **Gantt diagram** to visualize collisions and manually identify potential swaps.
