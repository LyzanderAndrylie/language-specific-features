import os
import re
from pathlib import Path

import pandas as pd

current_dir = Path(__file__).parent
data_dir = current_dir.parent / "all"

output_excel = current_dir / "combined_outputs.xlsx"

with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
    for subdir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)

                parent_dir = os.path.basename(subdir)
                base_name = os.path.splitext(file)[0]

                match = re.match(r"(seed_\d+_mult_\d+\.\d+)", base_name)
                if match:
                    base_name = match.group(1)

                excel_max_sheet_name_length = 31
                sheet_name = f"{parent_dir}_{base_name}"[:excel_max_sheet_name_length]

                df = pd.read_csv(file_path)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"All CSVs have been combined into {output_excel}")
