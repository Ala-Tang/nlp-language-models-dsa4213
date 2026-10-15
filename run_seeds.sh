set -e
cd /content/drive/MyDrive/dsa4213/assignment1
mv results results_seed42
mv checkpoints checkpoints_seed42
for s in 1 2 3; do
  sed -i -E "s/^(\s*seed:\s*)[0-9]+\s*$/\1$s/" config.yaml
  grep -Eq "^\s*seed:\s*$s\s*$" config.yaml || { echo "SEED NOT CHANGED, STOP"; exit 1; }
  echo "===== seed $s ====="
  python run_all.py
  mv results results_seed$s
  mv checkpoints checkpoints_seed$s
done
sed -i -E "s/^(\s*seed:\s*)[0-9]+\s*$/\142/" config.yaml
echo "ALL SEEDS DONE"
