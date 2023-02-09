#!/bin/bash
sbatch --partition=clara --job-name=test --time=2-00:00:00 --ntasks=1 --mem=20g --cpus-per-task=11