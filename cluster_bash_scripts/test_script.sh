#!/bin/bash
sbatch --partition=clara --job-name=test --time=2-00:00:00 --n-tasks=1 --mem=20g --cpus-per-task=11