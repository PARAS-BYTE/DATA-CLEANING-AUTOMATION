import pandas as pd
import numpy as np

def generate_sample_dirty_dataset(filepath: str = "sample_dirty_data.csv") -> str:
    """
    Generates a realistic messy dataset with various data quality issues:
    - Missing values (numeric and text)
    - Duplicate records
    - Inconsistent string casing and whitespaces
    - Currency / comma-separated numbers formatted as strings
    - Inconsistent date formats
    - Constant column
    - Mixed types
    """
    np.random.seed(42)
    
    data = {
        "Customer_ID": [
            "CUST-001", "CUST-002", "CUST-003", "CUST-004", "CUST-005",
            "CUST-006", "CUST-007", "CUST-008", "CUST-009", "CUST-010",
            "CUST-003", # Duplicate row ID
            "CUST-011", "CUST-012", "CUST-013", "CUST-014", "CUST-015"
        ],
        "Full_Name": [
            "  John Doe ", "Jane Smith", "Robert Brown", "Emily Davis", "Michael Wilson",
            "Sarah Taylor", "David Clark", "Jessica White", "James Harris", "Laura Martin",
            "Robert Brown", # Duplicate
            "Daniel Lewis", "Ashley Robinson", "Matthew Walker", "Amanda Hall", "Joshua Allen"
        ],
        "Age": [
            28, np.nan, 45, 32, np.nan,
            54, 23, 39, 41, np.nan,
            45, # Duplicate
            31, 29, 62, 35, 48
        ],
        "Annual_Income": [
            "$55,000", "$68,500", "$48,000", "$120,000", "$75,200",
            "$92,000", "$41,000", "$83,400", "$61,000", "$115,000",
            "$48,000", # Duplicate
            "$52,000", "$71,500", "$98,000", "$64,200", "$88,000"
        ],
        "Signup_Date": [
            "2022-01-15", "15/02/2022", "2022-03-10", "April 5, 2022", "2022-05-20",
            "2022-06-18", "07/19/2022", "2022-08-25", "Sept 12, 2022", "2022-10-01",
            "2022-03-10", # Duplicate
            "2022-11-14", "12/05/2022", "2023-01-08", "2023-02-14", "2023-03-22"
        ],
        "Account_Type": [
            "Premium", "standard", "PREMIUM", "Standard", "VIP",
            "Standard", "standard", "VIP", "Premium", "Standard",
            "PREMIUM", # Duplicate
            "standard", "VIP", "Premium", "Standard", "standard"
        ],
        "Credit_Score": [
            720, 680, np.nan, 790, 640,
            810, 590, 750, 705, np.nan,
            np.nan, # Duplicate
            695, 730, 805, 660, 740
        ],
        "System_Region": [
            "GLOBAL", "GLOBAL", "GLOBAL", "GLOBAL", "GLOBAL",
            "GLOBAL", "GLOBAL", "GLOBAL", "GLOBAL", "GLOBAL",
            "GLOBAL", # Constant column
            "GLOBAL", "GLOBAL", "GLOBAL", "GLOBAL", "GLOBAL"
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return filepath

if __name__ == "__main__":
    path = generate_sample_dirty_dataset()
    print(f"Sample dirty dataset generated at: {path}")
