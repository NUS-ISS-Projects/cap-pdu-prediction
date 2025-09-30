"""Chat routes for PDU data analysis and querying."""

from datetime import datetime, timedelta
import re
from flask import Blueprint, request, jsonify, current_app
from data.models import ChatRequest

chat_bp = Blueprint("chat", __name__)


def get_rag_system():
    """Get the RAG system from the Flask application context."""
    return current_app.rag_system


def parse_time_period(query):
    """Parse user query to extract time period."""
    query_lower = query.lower()

    # Check for specific patterns
    if re.search(r"24\s*hours?|past\s*day|today", query_lower):
        return "day", 1
    elif re.search(r"week|7\s*days?", query_lower):
        return "week", 1
    elif re.search(r"month|30\s*days?", query_lower):
        return "month", 1
    elif re.search(r"(\d+)\s*hours?", query_lower):
        hours = int(re.search(r"(\d+)\s*hours?", query_lower).group(1))
        return "hour", hours
    elif re.search(r"(\d+)\s*days?", query_lower):
        days = int(re.search(r"(\d+)\s*days?", query_lower).group(1))
        return "day", days
    elif re.search(r"(\d+)\s*weeks?", query_lower):
        weeks = int(re.search(r"(\d+)\s*weeks?", query_lower).group(1))
        return "week", weeks

    # Default to past week
    return "week", 1


def generate_analysis(data, time_unit, periods):
    """Generate meaningful analysis from the data."""
    if not data or "buckets" not in data:
        return "No data available for the requested time period."

    buckets = data["buckets"]
    total_packets = sum(bucket.get("totalPackets", 0) for bucket in buckets)

    if total_packets == 0:
        return (
            f"No PDU activity detected in the past {periods} {time_unit}(s). "
            "The system appears to be inactive during this period."
        )

    # Find peak activity period
    peak_bucket = max(buckets, key=lambda x: x.get("totalPackets", 0))
    peak_packets = peak_bucket.get("totalPackets", 0)

    # Calculate different PDU types
    entity_state = sum(bucket.get("entityStatePduCount", 0) for bucket in buckets)
    fire_events = sum(bucket.get("fireEventPduCount", 0) for bucket in buckets)
    collisions = sum(bucket.get("collisionPduCount", 0) for bucket in buckets)
    detonations = sum(bucket.get("detonationPduCount", 0) for bucket in buckets)

    # Generate time period description
    time_period = ""
    if time_unit == "hour":
        time_period = f"past {periods} hour(s)" if periods > 1 else "past hour"
    elif time_unit == "day":
        time_period = f"past {periods} day(s)" if periods > 1 else "past day"
    elif time_unit == "week":
        time_period = f"past {periods} week(s)" if periods > 1 else "past week"
    elif time_unit == "month":
        time_period = f"past {periods} month(s)" if periods > 1 else "past month"

    # Build analysis
    analysis = f"📊 **PDU Activity Analysis for {time_period}:**\n\n"
    analysis += f"• **Total PDU Packets:** {total_packets:,}\n"
    analysis += (
        f"• **Entity State PDUs:** {entity_state:,} ({entity_state/total_packets*100:.1f}%)\n"
        if total_packets > 0
        else ""
    )
    analysis += f"• **Fire Event PDUs:** {fire_events:,}\n" if fire_events > 0 else ""
    analysis += f"• **Collision PDUs:** {collisions:,}\n" if collisions > 0 else ""
    analysis += f"• **Detonation PDUs:** {detonations:,}\n" if detonations > 0 else ""

    if time_unit in ["week", "month"] and len(buckets) > 1:
        # Find peak period
        peak_desc = peak_bucket.get(
            "week", peak_bucket.get("date", peak_bucket.get("hour", "Unknown"))
        )
        analysis += f"\n• **Peak Activity:** {peak_packets:,} packets during {peak_desc}\n"

    # Add insights
    if entity_state > total_packets * 0.8:
        analysis += (
            "\n💡 **Insight:** High entity state activity suggests "
            "active simulation with many moving objects."
        )
    if fire_events > 0:
        analysis += (
            f"\n🔥 **Combat Activity:** {fire_events} fire events detected during this period."
        )
    if collisions > 0 or detonations > 0:
        analysis += (
            f"\n💥 **Impact Events:** {collisions + detonations} "
            "collision/detonation events recorded."
        )

    return analysis


@chat_bp.route("/chat", methods=["POST"])
@chat_bp.route("/api/prediction/chat", methods=["POST"])
def chat():
    """Handle chat requests with PDU data analysis."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        question = data.get("question")
        if not question:
            return jsonify({"error": "Question is required"}), 400

        req = ChatRequest(question=question, session_id=data.get("sessionId"))

        rag = get_rag_system()

        # Extract JWT token from request headers
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            rag.set_jwt_token(token)

        if not rag.jwt_token:
            return (
                jsonify(
                    {
                        "error": (
                            "Authentication required. Please ensure you have "
                            "valid access to the data acquisition service."
                        )
                    }
                ),
                401,
            )

        # Parse the query to understand what data is needed
        time_unit, periods = parse_time_period(req.question)

        # Calculate date range
        end_date = datetime.now().date()
        if time_unit == "hour":
            start_date = end_date
        elif time_unit == "day":
            start_date = end_date - timedelta(days=periods)
        elif time_unit == "week":
            start_date = end_date - timedelta(weeks=periods)
        elif time_unit == "month":
            start_date = end_date - timedelta(days=periods * 30)
        else:
            # Default to week if time_unit is unrecognized
            start_date = end_date - timedelta(weeks=1)

        # Fetch appropriate data
        try:
            historical_data = rag.fetch_historical_data(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), time_unit
            )

            # Generate meaningful analysis
            answer = generate_analysis(historical_data, time_unit, periods)

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error fetching data: {e}")
            answer = (
                "Unable to fetch data from the acquisition service. "
                "Please ensure the service is running and accessible."
            )

        return jsonify(
            {
                "answer": answer,
                "question": req.question,
                "sessionId": req.session_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@chat_bp.route("/chat/status", methods=["GET"])
@chat_bp.route("/api/prediction/chat/status", methods=["GET"])
def chat_status():
    """Get chat service status with authentication and cache information."""
    rag = get_rag_system()
    return jsonify(
        {
            "authenticated": bool(rag.jwt_token),
            "cache_summary": rag.cache.get_cache_summary(),
            "vector_store_ready": bool(rag.vector_store.texts),
        }
    )
