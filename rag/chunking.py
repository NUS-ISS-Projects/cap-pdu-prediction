"""Data chunking module for formatting PDU data into text chunks."""

from typing import List, Dict, Any


class DataChunker:
    """Formats PDU data into text chunks for vector storage."""

    @staticmethod
    def format_realtime_data(realtime_data: Dict[str, Any]) -> str:
        """Format real-time PDU data into a readable text chunk."""
        if not realtime_data:
            return "Real-time data: Not available"

        return f"""Real-time DIS PDU Data Summary:
- Total PDUs in last 60 seconds: {realtime_data.get('pdusInLastSixtySeconds', 0)}
- Average PDU rate per second: {realtime_data.get('averagePduRatePerSecondLastSixtySeconds', 0):.2f}
- Entity State PDUs: {realtime_data.get('entityStatePdusInLastSixtySeconds', 0)}
- Fire Event PDUs: {realtime_data.get('fireEventPdusInLastSixtySeconds', 0)}
- Collision PDUs: {realtime_data.get('collisionPdusInLastSixtySeconds', 0)}
- Detonation PDUs: {realtime_data.get('detonationPdusInLastSixtySeconds', 0)}
- Data PDUs: {realtime_data.get('dataPdusInLastSixtySeconds', 0)}"""

    @staticmethod
    def format_aggregated_data(data: Dict[str, Any], period_name: str) -> List[str]:
        """Format aggregated PDU data into text chunks."""
        chunks = []
        if not data or "buckets" not in data:
            chunks.append(f"{period_name}: No aggregated data available")
            return chunks

        time_unit = data.get("timeUnit", "unknown")
        buckets = data["buckets"]

        if time_unit == "hour":
            chunks.append(f"Hourly DIS PDU Data ({period_name}):")
            for bucket in buckets:
                hour = bucket.get("hour", "N/A")
                total = bucket.get("totalPackets", 0)
                if total > 0:
                    chunks.append(f"  Hour {hour}: {total} total PDUs")

        elif time_unit == "day":
            chunks.append(f"Daily DIS PDU Data ({period_name}):")
            for bucket in buckets:
                date = bucket.get("date", "N/A")
                total = bucket.get("totalPackets", 0)
                if total > 0:
                    chunks.append(f"  {date}: {total} total PDUs")

        elif time_unit == "week":
            chunks.append(f"Weekly DIS PDU Data ({period_name}):")
            for bucket in buckets:
                week = bucket.get("week", "N/A")
                total = bucket.get("totalPackets", 0)
                if total > 0:
                    chunks.append(f"  Week {week}: {total} total PDUs")

        return chunks

    @staticmethod
    def format_pdu_logs(logs_data: Dict[str, Any], period_name: str) -> List[str]:
        """Format PDU logs into text chunks."""
        chunks = []
        if not logs_data or "logs" not in logs_data:
            chunks.append(f"{period_name} PDU logs: No detailed logs available")
            return chunks

        logs = logs_data["logs"]
        total_count = len(logs)
        chunks.append(f"{period_name} PDU Logs: {total_count} individual PDU events recorded")

        pdu_type_counts = {}
        for log in logs:
            pdu_type = log.get("pduType", "Unknown")
            pdu_type_counts[pdu_type] = pdu_type_counts.get(pdu_type, 0) + 1

        if pdu_type_counts:
            chunks.append(f"PDU Type Distribution in {period_name}:")
            for pdu_type, count in sorted(pdu_type_counts.items()):
                chunks.append(f"  {pdu_type}: {count} PDUs")

        return chunks
