function ClassificationReport({ classificationReport }) {
  if (!classificationReport) return null;

  return (
    <div className="classification-report">
      <h3>Classification Details</h3>
      <div className="report-table-wrapper">
        <table className="report-table">
          <thead>
            <tr className="report-header">
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(classificationReport).map(([className, metrics]) => {
              // Exclude 'accuracy', 'macro avg', 'weighted avg' if they exist in the report
              if (['accuracy', 'macro avg', 'weighted avg'].includes(className)) return null;
              
              const isApproved = className.toLowerCase() === 'approved';
              return (
                <tr className="report-row" key={className}>
                  <td>
                    <span className={`class-badge ${isApproved ? 'approved' : 'rejected'}`}>
                      {className} {isApproved ? '(1)' : '(0)'}
                    </span>
                  </td>
                  <td><span className="metric-value">{metrics.precision || metrics.precision_approved || metrics.precision_rejected}</span></td>
                  <td><span className="metric-value">{metrics.recall || metrics.recall_approved || metrics.recall_rejected}</span></td>
                  <td><span className="metric-value">{metrics['f1-score'] || metrics.f1_score || metrics.f1_approved || metrics.f1_rejected}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ClassificationReport;
