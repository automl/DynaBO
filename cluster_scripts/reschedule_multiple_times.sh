#!/bin/bash
#SBATCH --job-name=dynamic_job_submission
#SBATCH --time=2-00:00:00  # Maximum runtime of 2 days
#SBATCH --output=dynamic_submission.log
#SBATCH --error=dynamic_submission.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Paths to the baseline and approach job scripts
BASELINE_SCRIPT="cluster_scripts/baseline-alloc.sh"
PRIOR_SCRIPT="cluster_scripts/prior-alloc.sh"

# Target number of jobs to maintain
TARGET_JOBS=300

# Total number of jobs to submit for each type
TOTAL_BASELINE_JOBS=1200
TOTAL_PRIOR_JOBS=1200

# Counters for submitted jobs
SUBMITTED_BASELINE_JOBS=0
SUBMITTED_PRIOR_JOBS=0

# Function to check if any jobs are scheduled or running
jobs_scheduled_or_running() {
    local job_count
    job_count=$(squeue -u $USER | wc -l)
    [ "$job_count" -gt 3 ]
}

# Infinite loop to monitor and submit jobs
while true; do
    # Check if jobs are currently scheduled or running
    if (( SUBMITTED_BASELINE_JOBS >= TOTAL_BASELINE_JOBS && SUBMITTED_PRIOR_JOBS >= TOTAL_PRIOR_JOBS )); then
        echo "All jobs submitted: $SUBMITTED_BASELINE_JOBS baseline jobs and $SUBMITTED_PRIOR_JOBS prior jobs."
        exit 0
    fi

    if jobs_scheduled_or_running; then
        echo "Jobs are already scheduled or running. No jobs submitted."
    else
        echo "No jobs currently scheduled or running. Submitting jobs..."
        sbatch $BASELINE_SCRIPT
        SUBMITTED_BASELINE_JOBS=$(( SUBMITTED_BASELINE_JOBS + 150 ))
        sbatch $PRIOR_SCRIPT
        SUBMITTED_PRIOR_JOBS=$(( SUBMITTED_PRIOR_JOBS + 150 ))
    fi

    # Sleep for 2 minutes before the next check
    echo "Pausing for 2 minutes..."
    sleep 120

done
