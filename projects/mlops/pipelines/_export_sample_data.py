import pandas as pd
import numpy as np
import random
df = pd.read_parquet('/Users/danherman/Desktop/enterprise_clinical_copilot/projects/mlops/pipelines/sample_data.parquet')
for col in df.columns:
    if df[col].dtype.name != 'category' and col != 'split':
        # Add tiny noise to numerical columns if they are all identical so correlation doesn't divide by 0
        if df[col].std() == 0:
            df[col] = df[col] + np.random.normal(0, 0.001, size=len(df))
df.to_parquet('/Users/danherman/Desktop/enterprise_clinical_copilot/projects/mlops/pipelines/sample_data.parquet')
print("Added noise to 0 variance numerical columns")
