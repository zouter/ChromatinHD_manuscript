import hashlib

params_peakcounts = []
params_peakcounts.append({"peakcaller": "macs2_leiden_0.1_merged", "predictor": "linear", "label": "peaks_main"})
params_peakcounts.append({"peakcaller": "encode_screen", "predictor": "linear", "label": "encode_screen_linear"})
# params_peakcounts.append({"peakcaller": "macs2_leiden_0.1_merged", "predictor": "rf", "label": "peaks_rf"})
# params_peakcounts.append({"peakcaller": "rolling_500", "predictor": "linear"})
# params_peakcounts.append({"peakcaller":"macs2_leiden_0.1_merged", "predictor":"lasso"})
# params_peakcounts.append({"peakcaller": "rolling_500", "predictor": "lasso", "label": "rolling_500_lasso"})
params_peakcounts.append({"peakcaller": "encode_screen", "predictor": "lasso", "label": "encode_screen_lasso"})

for dt in [0, 0.05, 0.1, 0.2]:
    params_peakcounts.append(
        {"peakcaller": "macs2_leiden_0.1_merged", "predictor": "linear", "label": f"dt{dt}/peaks_main", "dt": dt}
    )

params_peakcounts = {
    hashlib.md5(str({k: v2 for k, v2 in v.items() if k not in ["label"]}).encode()).hexdigest(): v
    for v in params_peakcounts
}
