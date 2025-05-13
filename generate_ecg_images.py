import os
import matplotlib.pyplot as plt
from helper_code import find_records, load_signals

def generate_ecg_image(record_path, save_path):
    signal, fields = load_signals(record_path)
    fig, axs = plt.subplots(6, 2, figsize=(10, 8))
    axs = axs.flatten()

    for i in range(signal.shape[1]):
        axs[i].plot(signal[:, i], linewidth=0.5)
        axs[i].set_title(fields['sig_name'][i])
        axs[i].axis('off')

    for i in range(signal.shape[1], 12):
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def generate_all_images(data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    records = find_records(data_folder)

    for record in records:
        base_name = os.path.basename(record)
        save_path = os.path.join(output_folder, f"{base_name}.png")
        generate_ecg_image(os.path.join(data_folder, record), save_path)

if __name__ == '__main__':
    # === Train data ===
    # generate_all_images('data_train/ptbxl_output', 'ecg_images/train/ptbxl')
    # generate_all_images('data_train/samitrop_output', 'ecg_images/train/samitrop')

    # === Test data ===
    generate_all_images('data_test/ptbxl_output', 'ecg_images/test/ptbxl')
    # generate_all_images('data_test/samitrop_output', 'ecg_images/test/samitrop')



# import os
# import shutil

# def get_folder_prefix(record_id):
#     """Extract folder prefix based on thousands grouping"""
#     num = int(record_id)
#     group = (num // 1000) * 1000
#     return f"{group:05d}"

# def migrate_images(src_root="ecg_images/test/ptbxl", dst_root="data_test/ptbxl_output"):
#     if not os.path.exists(dst_root):
#         os.makedirs(dst_root)

#     for file_name in os.listdir(src_root):
#         if not file_name.endswith("_hr.png"):
#             continue

#         record_id = file_name.replace("_hr.png", "")  # remove suffix
#         folder_prefix = get_folder_prefix(record_id)

#         src_path = os.path.join(src_root, file_name)
#         dst_folder = os.path.join(dst_root, folder_prefix)
#         dst_path = os.path.join(dst_folder, file_name)

#         os.makedirs(dst_folder, exist_ok=True)
#         shutil.move(src_path, dst_path)
#         print(f"Moved {src_path} -> {dst_path}")

# if __name__ == "__main__":
#     migrate_images()