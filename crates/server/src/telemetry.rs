//! Best-effort Linux/Android system telemetry for `/v1/info`.

use std::fs;

/// Memory and thermal telemetry read from `/proc/meminfo` and `/sys/class/thermal`.
#[derive(Debug, Clone, Default)]
pub struct SystemTelemetry {
    pub mem_total_mb: Option<u64>,
    pub mem_available_mb: Option<u64>,
    pub thermal_c: Option<f32>,
}

/// Reads current telemetry; any source that is missing or unreadable is left `None`.
pub fn read_telemetry() -> SystemTelemetry {
    let (mem_total_mb, mem_available_mb) = fs::read_to_string("/proc/meminfo")
        .map(|s| parse_meminfo(&s))
        .unwrap_or((None, None));

    let thermal_c = read_max_thermal_c();

    SystemTelemetry {
        mem_total_mb,
        mem_available_mb,
        thermal_c,
    }
}

/// Extracts `(MemTotal, MemAvailable)` in MiB from `/proc/meminfo` contents.
pub fn parse_meminfo(contents: &str) -> (Option<u64>, Option<u64>) {
    let mut total_kb = None;
    let mut avail_kb = None;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = parse_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = parse_kb(rest);
        }
    }
    (total_kb.map(|k| k / 1024), avail_kb.map(|k| k / 1024))
}

fn parse_kb(rest: &str) -> Option<u64> {
    rest.split_whitespace().next()?.parse::<u64>().ok()
}

fn read_max_thermal_c() -> Option<f32> {
    let entries = fs::read_dir("/sys/class/thermal").ok()?;
    let mut max_c: Option<f32> = None;

    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("thermal_zone") {
            continue;
        }
        let temp_path = entry.path().join("temp");
        let Ok(raw) = fs::read_to_string(temp_path) else {
            continue;
        };
        let Ok(milli_c) = raw.trim().parse::<i64>() else {
            continue;
        };
        // Ignore bogus/disconnected sensors: they report negatives or values like 255 °C;
        // nothing on a working phone exceeds 150 °C (150_000 millidegrees).
        if !(0..=150_000).contains(&milli_c) {
            continue;
        }
        let c = (milli_c as f32) / 1000.0;
        max_c = Some(max_c.map_or(c, |cur| cur.max(c)));
    }

    max_c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_proc_meminfo_values() {
        let sample = "\
MemTotal:       16384000 kB
MemFree:         2048000 kB
MemAvailable:    8192000 kB
";
        assert_eq!(parse_meminfo(sample), (Some(16000), Some(8000)));
    }
}
