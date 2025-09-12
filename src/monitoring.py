# Modelo de integração com Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_drift_report(reference, current):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("reports/data_drift_report.html")
