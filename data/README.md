# Healthcare analytics datasets

These are the supplied CSV datasets used by the Text-to-SQL analytics pipeline.

- `patients.csv`: 12,000 patient records
- `prescriptions.csv`: 25,000 prescription records
- `visits.csv`: 15,000 visit records

DuckDB loads these files as the `patients`, `prescriptions`, and `visits` tables when the Flask app starts.

## Relationships

- `patients.patient_id` joins to `visits.patient_id`
- `patients.patient_id` joins to `prescriptions.patient_id`
- `visits.visit_id` joins to `prescriptions.visit_id`

## Example analytics questions

- How many patients are in the dataset?
- What is the average age of patients?
- How many patients are registered in each hospital location?
- How many patients have each chronic condition?
- What are the top 10 most prescribed drugs?
- What is the total prescription cost by drug?
- How many visits occurred each year?
- What is the average visit cost by doctor speciality?
- How many patients were readmitted within 30 days?
- Which diagnoses have the highest prescription counts?

The datasets contain identifiable-looking fields such as names, phone numbers and emails. Keep them local or use properly anonymized copies before deploying or sharing the project. The sample values appear synthetic, but treat the files as sensitive unless their provenance confirms otherwise.
