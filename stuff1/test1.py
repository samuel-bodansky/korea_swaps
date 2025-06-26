import blpapi
from datetime import datetime


def get_historical_data(ticker, field, start_date, end_date):
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost("localhost")
    sessionOptions.setServerPort(8194)

    session = blpapi.Session(sessionOptions)
    if not session.start():
        print("Failed to start session.")
        return

    if not session.openService("//blp/refdata"):
        print("Failed to open //blp/refdata service.")
        session.stop()
        return

    refDataService = session.getService("//blp/refdata")
    request = refDataService.createRequest("HistoricalDataRequest")

    request.append("securities", ticker)
    request.append("fields", field)
    request.set("startDate", start_date)  # YYYYMMDD
    request.set("endDate", end_date)  # YYYYMMDD
    request.set("periodicitySelection", "DAILY")  # DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY

    print(f"Sending historical data request for {ticker} - {field} from {start_date} to {end_date}...")
    cid = blpapi.CorrelationId()
    session.sendRequest(request, correlationId=cid)

    historical_data = []

    while (True):
        event = session.nextEvent(100)
        if event.eventType() == blpapi.Event.RESPONSE or \
                event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
            for msg in event:
                # print(f"Received message: {msg}") # Uncomment to see full message structure
                securityData = msg.getElement("securityData")
                fieldDataArray = securityData.getElement("fieldData")

                for fieldData in fieldDataArray.values():
                    date = fieldData.getElementAsDatetime("date").strftime("%Y-%m-%d")
                    value = fieldData.getElementAsFloat(field)
                    historical_data.append({"date": date, field: value})

            if event.eventType() == blpapi.Event.RESPONSE:
                break

    session.stop()
    return historical_data


if __name__ == "__main__":
    # Example: Get historical closing prices for SPY for January 2024
    spy_history = get_historical_data("SPY US Equity", "PX_LAST", "20240101", "20240131")
    if spy_history:
        print("\nHistorical Data for SPY US Equity:")
        for entry in spy_history:
            print(entry)