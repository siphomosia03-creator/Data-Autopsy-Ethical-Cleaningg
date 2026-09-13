import os
import difflib
import pandas as pd
import numpy as np


class DataCleaningError(Exception):
    """Raised when data validation fails during cleaning."""
    pass


class DataCleaner:
    """Ethical data cleaning for South African fintech customer data."""

    def __init__(self, df):
        self.df = df.copy()
        self.ethical_notes = []
        self.cleaning_log = []
        self.region_changes = 0
        self.rows_cleaned = len(self.df)

    def validate_income(self):
        """Convert income to numeric, handle negatives and cap extreme outliers."""

        if "income" not in self.df.columns:
            raise DataCleaningError("Required column 'income' is missing.")

        if "township_flag" not in self.df.columns:
            raise DataCleaningError("Required column 'township_flag' is missing.")

        self.df["income"] = (
            self.df["income"]
            .astype("string")
            .str.replace("R", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        self.df["income"] = pd.to_numeric(
            self.df["income"],
            errors="coerce"
        )

        negative_income_count = (self.df["income"] < 0).sum()

        if negative_income_count > 0:
            self.df.loc[self.df["income"] < 0, "income"] = np.nan
            self.cleaning_log.append(
                f"Converted {negative_income_count} negative income values to missing"
            )

        valid_income = self.df["income"].dropna()

        if len(valid_income) == 0:
            raise DataCleaningError("No valid income values remain after conversion.")

        income_cap = valid_income.quantile(0.99)
        outlier_count = (self.df["income"] > income_cap).sum()

        self.df["income"] = self.df["income"].clip(upper=income_cap)

        self.cleaning_log.append(
            f"Capped {outlier_count} income outliers at the 99th percentile "
            f"(R{income_cap:.2f})"
        )

        township_mask = self.df["township_flag"] == 1
        non_township_mask = self.df["township_flag"] == 0

        township_missing_rate = (
            self.df.loc[township_mask, "income"].isna().mean()
            if township_mask.sum() > 0
            else 0
        )

        non_township_missing_rate = (
            self.df.loc[non_township_mask, "income"].isna().mean()
            if non_township_mask.sum() > 0
            else 0
        )

        if township_missing_rate > non_township_missing_rate:
            difference = (
                township_missing_rate - non_township_missing_rate
            ) * 100

            self.ethical_notes.append(
                f"WARNING: Township applicants show {difference:.1f} percentage "
                f"points higher income missingness than non-township applicants. "
                f"This may indicate form accessibility or representation issues."
            )
        else:
            self.ethical_notes.append(
                "Income missingness was assessed by township status and retained "
                "as an important representation and data-quality signal."
            )

        self.cleaning_log.append("Income validation completed")

        return self.df

    def standardize_regions(self):
        """Map regional variants to standardized names."""

        if "region" not in self.df.columns:
            raise DataCleaningError("Required column 'region' is missing.")

        region_map = {
            "JHB": "Johannesburg",
            "Joburg": "Johannesburg",
            "Johburg": "Johannesburg",
            "Johannesburg": "Johannesburg",
            "CPT": "Cape Town",
            "Capetown": "Cape Town",
            "Cape Town": "Cape Town",
            "DBN": "Durban",
            "eThekwini": "Durban",
            "Durban": "Durban",
            "Pta": "Pretoria",
            "Tshwane": "Pretoria",
            "Pretoria": "Pretoria",
            "PE": "Gqeberha",
            "Nelson Mandela Bay": "Gqeberha",
            "Gqeberha": "Gqeberha",
            "Western Cape": "Cape Town",
            "Gauteng": "Johannesburg"
        }

        original_regions = self.df["region"].copy()

        self.df["region"] = (
            self.df["region"]
            .astype("string")
            .str.strip()
        )

        self.df["region"] = self.df["region"].replace(region_map)

        known_regions = [
            "Johannesburg",
            "Cape Town",
            "Durban",
            "Pretoria",
            "Gqeberha"
        ]

        unmatched_mask = (
            self.df["region"].notna()
            & ~self.df["region"].isin(known_regions)
        )

        for index in self.df.index[unmatched_mask]:
            value = self.df.at[index, "region"]

            match = difflib.get_close_matches(
                value,
                known_regions,
                n=1,
                cutoff=0.75
            )

            if match:
                self.df.at[index, "region"] = match[0]

        self.region_changes = (
            original_regions.fillna("<MISSING>").astype(str)
            != self.df["region"].fillna("<MISSING>").astype(str)
        ).sum()

        self.cleaning_log.append(
            f"Fixed {self.region_changes} region variants"
        )

        return self.df

    def create_missing_indicators(self):
        """Create binary flags for missing critical values."""

        for column in ["income", "township_flag"]:
            if column not in self.df.columns:
                raise DataCleaningError(
                    f"Required column '{column}' is missing."
                )

        self.df["income_missing"] = (
            self.df["income"].isna().astype(int)
        )

        self.df["township_flag_missing"] = (
            self.df["township_flag"].isna().astype(int)
        )

        income_missing_count = self.df["income_missing"].sum()
        township_missing_count = self.df["township_flag_missing"].sum()

        self.cleaning_log.append(
            f"Created income_missing indicator ({income_missing_count} missing)"
        )

        self.cleaning_log.append(
            f"Created township_flag_missing indicator "
            f"({township_missing_count} missing)"
        )

        self.ethical_notes.append(
            "Missing-value indicators preserve information about where data "
            "is unavailable instead of silently treating missing values as "
            "ordinary observations."
        )

        return self.df

    def to_csv(self, output_path):
        """Export cleaned data using UTF-8 encoding and datetime formatting."""

        if self.df.empty:
            raise DataCleaningError(
                "Cannot export an empty cleaned dataset."
            )

        output_directory = os.path.dirname(output_path)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        self.df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d %H:%M:%S"
        )

        self.cleaning_log.append(
            f"Exported cleaned data to {output_path}"
        )

    def __str__(self):
        return (
            f"Cleaned {self.rows_cleaned:,} rows | "
            f"Fixed {self.region_changes:,} region variants"
        )

    def run_full_cleaning(self):
        """Execute all cleaning steps in ethical sequence."""

        try:
            self.validate_income()
            self.standardize_regions()
            self.create_missing_indicators()

            self.cleaning_log.append("Full cleaning completed")

            return self.df

        except DataCleaningError:
            raise

        except Exception as e:
            raise DataCleaningError(
                f"Cleaning failed: {str(e)}"
            ) from e


if __name__ == "__main__":
    input_path = "data/raw/customer_loans_q1_2024.csv"
    output_path = "data/processed/cleaned_customers.csv"

    try:
        if not os.path.exists(input_path):
            raise DataCleaningError(
                f"Raw dataset not found: {input_path}"
            )

        df = pd.read_csv(
            input_path,
            dtype={
                "customer_id": "string",
                "region": "string",
                "income": "string",
                "loan_purpose": "string"
            }
        )

        cleaner = DataCleaner(df)
        cleaner.run_full_cleaning()
        cleaner.to_csv(output_path)

        print(cleaner)

        print("\nCleaning log:")
        for entry in cleaner.cleaning_log:
            print(f"- {entry}")

        print("\nEthical notes:")
        for note in cleaner.ethical_notes:
            print(f"- {note}")

        print(f"\nOutput: {output_path}")

    except DataCleaningError as e:
        print(f"DataCleaningError: {e}")
        raise
