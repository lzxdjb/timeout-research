export SWE_AGENT_EXECUTION_URLS="http://10.248.100.94:18080,http://10.248.100.208:18080,http://10.248.100.205:18080,http://10.248.100.203:18080,http://10.248.100.193:18080,http://10.248.100.87:18080,http://10.248.100.14:18080"


: "${SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS:=5}"

IFS=',' read -r -a execution_urls <<< "$SWE_AGENT_EXECUTION_URLS"
failed=0

for execution_url in "${execution_urls[@]}"; do
    execution_url="${execution_url%/}"

    if [[ -z "$execution_url" ]]; then
        echo "ERROR: empty endpoint"
        failed=1
        continue
    fi

    echo "Checking: $execution_url"

    health_body="$(
        curl --noproxy '*' \
             --fail --silent --show-error \
             --max-time "$SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS" \
             "$execution_url/health"
    )"
    curl_rc=$?

    if (( curl_rc != 0 )); then
        echo "FAIL: unable to connect to $execution_url (curl exit $curl_rc)" >&2
        failed=1
        continue
    fi

    health_compact="$(printf '%s' "$health_body" | tr -d '[:space:]')"

    if [[ "$health_compact" != *'"ok":true'* ]]; then
        echo "FAIL: unhealthy response from $execution_url" >&2
        printf 'Health response: %s\n' "$health_body" >&2
        failed=1
        continue
    fi

    echo "OK: $execution_url"
done

if (( failed != 0 )); then
    echo "One or more SWE execution pods failed health checks." >&2
    exit 1
fi