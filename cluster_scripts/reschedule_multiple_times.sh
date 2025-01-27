#!/bin/bash
#SBATCH --job-name=dynamic_job_submission
#SBATCH --time=2-00:00:00  # Maximum runtime of 2 days
#SBATCH --output=dynamic_submission.log
#SBATCH --error=dynamic_submission.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

DYNABO_RUN_SCRIPT="cluster_scripts/dynabo-alloc.sh"
PIBO_RUN_SCRIPT="cluster_scripts/pibo_alloc.sh"

TOTAL_JOBS=3300
N_PARALLEL_JOBS=300

# Initialize counters
SUBMITTED_JOBS=0

# Function to check if any jobs are scheduled or running
jobs_scheduled_or_running() {
    local job_count
    job_count=$(squeue -u $USER | wc -l)
    [ "$job_count" -gt 2 ]
}

# Infinite loop to monitor and submit jobs
while true; do
    # Check if jobs are currently scheduled or running
    if (( SUBMITTED_JOBS >= TOTAL_JOBS )); then
        echo "$(date) - All jobs submitted. Exiting..."
        exit 0
    fi

    if jobs_scheduled_or_running; then
        echo "$(date) - Jobs are already scheduled or running. No jobs submitted."
    else
        echo "$(date) - No jobs currently scheduled or running. Submitting jobs..."
        #sbatch $DYNABO_RUN_SCRIPT
        sbatch $PIBO_RUN_SCRIPT
        sbatch
        SUBMITTED_JOBS=$(( SUBMITTED_JOBS + N_PARALLEL_JOBS ))
        echo "$(date) - Submitted $N_PARALLEL_JOBS jobs. Total submitted: $SUBMITTED_JOBS."
    fi

    # Sleep for 2 minutes before the next check
    echo "$(date) - Pausing for 2 minutes..."
    sleep 120
done
