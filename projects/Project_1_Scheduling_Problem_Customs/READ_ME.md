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



**Presentation of files:** 
  - Datas : 
    - Arrivals_HD.xlsx : input with the schedule of arrivals
    - Average_time.xlsx : input of average times to compute the passenger-minutes
  - Interface:
    - run.bat : the launcher of interface
    - Interface.py : this file handles the front-end of the interface for the user
    - Escondido.py : this file handles the structure of interface coordinating the function while the interface is used
  - Core files:
    - Process.py : this file handles the gathering and formating of data
    - Optimizer.py : this file handles the solving part 
    - Resultats.py : This files organizes the diferent oprimizations and return mains results
    - Viz.ipynb : a jupyter notebook to by-pass the interface that could be long sometimes 
