import pandas as pd

tests = {
    # 'PACS': ['art_painting', 'cartoon', 'photo'],
    # 'PACS-photo' : ['art_painting', 'cartoon', 'sketch'],
    # 'PACS-cartoon': ['art_painting', 'photo', 'sketch'],
    # 'PACS-art_painting': ['cartoon'],
    # 'terra-incognita-38': ['location_43', 'location_46', 'location_100'],
    # 'terra-incognita-43': ['location_38', 'location_46', 'location_100'],
    'terra-incognita-43': ['location_46'],
    # 'terra-incognita-46': ['location_38', 'location_43', 'location_100'],
    # 'terra-incognita-100': ['location_38', 'location_43', 'location_46'],
    # 'camelyon17': ['hospital1', 'hospital2']
    # 'PACS-set2',
    # 'fmow-set2',
    # 'camelyon17-set2'
}

# for task_name in tests:
#     csv_path = f"metrics/{task_name}_model_0_kl_200_data.csv"
#     df = pd.read_csv(csv_path, comment="#")
#     dst_csv_path = f'metrics/{task_name}_model_0_kl_200_data.csv'
#     dst_df = pd.read_csv(dst_csv_path, comment="#")
#     dst_df[f"mcs_approx"] = df[f"mcs_approx"].values
#     dst_df[f"laplacian_spectral_dist"] = df[f"laplacian_spectral_dist"].values
#     dst_df[f"netlsd_dist"] = df[f"netlsd_dist"].values
#     dst_df[f"motif_counts_dist"] = df[f"motif_counts_dist"].values
#     dst_df[f"degree_distribution_dist"] = df[f"degree_distribution_dist"].values
#
#     dst_df.to_csv(dst_csv_path)

for task_name, domain_names in tests.items():
    csv_path = f"../vit-spurious-robustness/output/{task_name}_sweep_results_new.csv"
    df = pd.read_csv(csv_path)
    for domain in domain_names:
        dst_csv_path = f'metrics/{task_name}_model_{domain}_kl_data.csv'
        dst_df = pd.read_csv(dst_csv_path)
        dst_df[f"test_rankme_id"] = df[f"test_rankme_id"].values
        dst_df[f"test_alphaReQ_id"] = df[f"test_alphaReQ_id"].values
        dst_df[f"val_id_acc"] = df[f"val_id_acc"].values
        dst_df[f"val_id_f1"] = df[f"val_id_f1"].values
        dst_df[f"test_rankme_{domain}"] = df[f"test_rankme_{domain}"].values
        dst_df[f"test_alphaReQ_{domain}"] = df[f"test_alphaReQ_{domain}"].values
        dst_df[f"test_ATC_{domain}"] = df[f"test_ATC_{domain}"].values
        dst_df[f"test_EMD_{domain}"] = df[f"test_EMD_{domain}"].values
        dst_df[f"test_AC_{domain}"] = df[f"test_AC_{domain}"].values
        dst_df[f"test_ANE_{domain}"] = df[f"test_ANE_{domain}"].values
        dst_df[f"test_acc_{domain}"] = df[f"test_acc_{domain}"].values
        dst_df[f"test_f1_{domain}"] = df[f"test_f1_{domain}"].values
        dst_df[f"sharpness"] = df[f"sharpness"].values

        dst_df.to_csv(dst_csv_path)

# for task_name in tests:
#     csv_path = f"../vit-spurious-robustness/output/{task_name}_sweep_results_new.csv"
#     df = pd.read_csv(csv_path)
#     dst_csv_path = f'metrics/{task_name}_model_0_kl_200_data.csv'
#     dst_df = pd.read_csv(dst_csv_path)
#     dst_df[f"test_ATC_id"] = df[f"test_ATC_id"].values
#
#     dst_df[f"test_rankme_{domain}"] = df[f"test_rankme_{domain}"].values
#     dst_df[f"test_alphaReQ_{domain}"] = df[f"test_alphaReQ_{domain}"].values
#
#     dst_df.to_csv(dst_csv_path)
