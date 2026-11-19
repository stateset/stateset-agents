python -m stateset_agents.cli weight-edit teach examples/fanout/weight_edit/fact-1.json \
  --out "$scratch/edit" --metrics "$out/metrics.json" --threads 2 --report-only \
  --set revision=c1899de289a04d12100db370d81485cdf75e47ca \
  --set "sites=[{\"module\":\"model.layers.$parameter.mlp.down_proj\",\"key\":\"subject\"}]"
cp "$scratch/edit/recipe_report.json" "$out/result.json"
