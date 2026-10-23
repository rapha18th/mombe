# Pull the latest job scripts and start JOB in the background on this hub.
# The SSM command prepends: JOB=<name>; and optional env assignments.
export PATH=$PATH:/snap/bin
cd /opt/cool_equiv
aws s3 cp s3://mombe-967170427068-eu-west-1/jobs/1d/tools/ code/tools/cool_equiv/ --recursive --region eu-west-1 --only-show-errors
cp code/tools/cool_equiv/run_env.sh run_env.sh && chmod +x run_env.sh
mkdir -p out
nohup bash code/tools/cool_equiv/$JOB.sh > out/$JOB.$(date +%H%M%S).log 2>&1 &
echo "started $JOB pid $!"
