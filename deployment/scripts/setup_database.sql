-- Steel Defect Detection - Database Setup Script
-- Creates tables for storing inspection results on edge device

-- Create inspections table
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT NOT NULL,
    result TEXT NOT NULL,  -- PASS, FAIL, ERROR
    confidence REAL,
    processing_time REAL,
    trigger_method TEXT,   -- manual, auto, api
    metadata TEXT          -- JSON string for additional data
);

-- Create defects table
CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    area REAL,
    severity TEXT,         -- low, medium, high
    FOREIGN KEY (inspection_id) REFERENCES inspections(id)
);

-- Create system_logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    level TEXT NOT NULL,   -- DEBUG, INFO, WARNING, ERROR
    message TEXT NOT NULL,
    component TEXT,        -- camera, ai_model, hardware, etc.
    error_code TEXT
);

-- Create system_status table
CREATE TABLE IF NOT EXISTS system_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    cpu_usage REAL,
    memory_usage REAL,
    disk_usage REAL,
    temperature REAL,
    uptime INTEGER,
    camera_status TEXT,
    model_status TEXT,
    hardware_status TEXT
);

-- Create configuration_history table
CREATE TABLE IF NOT EXISTS configuration_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    config_key TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by TEXT
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    alert_type TEXT NOT NULL,  -- defect, system_error, maintenance
    severity TEXT NOT NULL,     -- low, medium, high, critical
    message TEXT NOT NULL,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at DATETIME,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at DATETIME
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp);
CREATE INDEX IF NOT EXISTS idx_inspections_result ON inspections(result);
CREATE INDEX IF NOT EXISTS idx_defects_inspection_id ON defects(inspection_id);
CREATE INDEX IF NOT EXISTS idx_defects_class_name ON defects(class_name);
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);

-- Insert initial system status
INSERT INTO system_status (cpu_usage, memory_usage, disk_usage, temperature, uptime, camera_status, model_status, hardware_status)
VALUES (0.0, 0.0, 0.0, 0.0, 0, 'initializing', 'initializing', 'initializing');

-- Insert initial configuration
INSERT INTO configuration_history (config_key, old_value, new_value, changed_by)
VALUES ('system_initialized', NULL, 'true', 'setup_script');

-- Create views for common queries
CREATE VIEW IF NOT EXISTS daily_inspection_summary AS
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_inspections,
    SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) as passed,
    SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END) as failed,
    SUM(CASE WHEN result = 'ERROR' THEN 1 ELSE 0 END) as errors,
    AVG(processing_time) as avg_processing_time
FROM inspections
GROUP BY DATE(timestamp)
ORDER BY date DESC;

CREATE VIEW IF NOT EXISTS defect_summary AS
SELECT 
    class_name,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    AVG(area) as avg_area,
    COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_severity,
    COUNT(CASE WHEN severity = 'medium' THEN 1 END) as medium_severity,
    COUNT(CASE WHEN severity = 'low' THEN 1 END) as low_severity
FROM defects
GROUP BY class_name
ORDER BY count DESC;

CREATE VIEW IF NOT EXISTS recent_alerts AS
SELECT 
    timestamp,
    alert_type,
    severity,
    message,
    acknowledged,
    acknowledged_by,
    resolved
FROM alerts
WHERE timestamp >= datetime('now', '-24 hours')
ORDER BY timestamp DESC;

-- Create trigger for automatic cleanup of old records
CREATE TRIGGER IF NOT EXISTS cleanup_old_inspections
AFTER INSERT ON inspections
WHEN (SELECT COUNT(*) FROM inspections) > 10000
BEGIN
    DELETE FROM inspections 
    WHERE id IN (
        SELECT id FROM inspections 
        ORDER BY timestamp ASC 
        LIMIT (SELECT COUNT(*) FROM inspections) - 10000
    );
END;

-- Create trigger for automatic alert generation on defects
CREATE TRIGGER IF NOT EXISTS create_defect_alert
AFTER INSERT ON defects
WHEN NEW.severity IN ('high', 'critical')
BEGIN
    INSERT INTO alerts (alert_type, severity, message)
    VALUES ('defect', NEW.severity, 
            'High severity ' || NEW.class_name || ' defect detected with confidence ' || NEW.confidence);
END;

-- Create trigger for logging configuration changes
CREATE TRIGGER IF NOT EXISTS log_config_changes
AFTER INSERT ON configuration_history
BEGIN
    INSERT INTO system_logs (level, message, component)
    VALUES ('INFO', 'Configuration changed: ' || NEW.config_key || ' = ' || NEW.new_value, 'configuration');
END;
