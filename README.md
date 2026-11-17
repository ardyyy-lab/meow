# TalentFlow Analytics (Python version)

A web app that models monthly recruiting revenue and interview capacity.

- **Placement planner** - Linear Programming (SciPy / HiGHS) with sensitivity analysis. Any number of job types.
  - **Profit gained and cost saved** - compares the best plan with the hires you make today.
  - **Solutions for the bottlenecks** - says which limits to raise, how much, what it costs, and shows the best plan after the fix.
- **Interview queue** - M/M/s queueing model, with cost saved against your current team and ways to cut the wait.

## Files

| File | What it is |
|---|---|
| `app.py` | The web page (Streamlit). |
| `model.py` | The math: LP, sensitivity analysis, queueing. |
| `test_model.py` | Checks the math against your Excel results. |
| `requirements.txt` | The Python libraries the app needs. |
| `.streamlit/config.toml` | The app's colors. |

## What's new: profit gained, cost saved, bottleneck solutions

**Placement planner**
- A new optional **Current hires** column in the job table. Fill it in to see profit gained per month and per year, fees gained, and cost per hire saved by switching to the best plan.
- A new **Fix the bottlenecks** box. Enter the most extra recruiter hours, candidates and sourcing budget you could add, and what each costs. The planner adds capacity only where the extra fees beat the extra cost, then shows what to add, the cost, the profit gained, the new plan and what still holds it back.

**Interview queue**
- A new **Interviewers you have today** input. Cost saved is measured against that team, with the pay change and the lost-candidate cost saved shown separately.
- A new **Solutions for the wait** table: more interviewers, faster interviews, or spreading out arrivals.

The cost figures in both tabs are placeholders. Replace them with your own numbers. `test_model.py` checks the new results too.

## Run it on your computer

1. Install Python 3.10 or newer from https://www.python.org/downloads/ (on Windows, tick "Add Python to PATH").
2. Open Terminal (Mac) or Command Prompt (Windows) and go to this folder:
   ```
   cd path/to/talentflow_python
   ```
3. Create a private environment and turn it on:
   - Mac / Linux: `python3 -m venv venv` then `source venv/bin/activate`
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
4. Install the libraries (one time): `pip install -r requirements.txt`
5. (Optional) Check the math: `python test_model.py`
6. Start the app: `streamlit run app.py`

Your browser opens at http://localhost:8501. Press `Ctrl + C` in the terminal to stop it.

Next time, only repeat steps 2, the "turn it on" part of step 3, and step 6.

## Put it online (free)

1. Create a GitHub account and a new public repository. Upload all the files in this folder.
2. Go to https://share.streamlit.io, sign in with GitHub, select **Create app**, pick your repository, and set the main file to `app.py`.
3. Select **Deploy**. You get a public link for your portfolio. Free apps go to sleep when nobody uses them, so the first load can take a minute.

## Troubleshooting

- `command not found: streamlit` - the environment is not on. Repeat the "turn it on" step.
- `ModuleNotFoundError` - run `pip install -r requirements.txt` again with the environment on.
- Port already in use - run `streamlit run app.py --server.port 8502`.
