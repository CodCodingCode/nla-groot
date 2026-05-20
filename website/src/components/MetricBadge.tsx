interface Props {
  verdict: "PASS" | "WARN" | "FAIL" | "NA";
}

export default function MetricBadge({ verdict }: Props) {
  const cls =
    verdict === "PASS"
      ? "badge pass"
      : verdict === "WARN"
        ? "badge warn"
        : verdict === "FAIL"
          ? "badge fail"
          : "badge";
  return <span className={cls}>{verdict}</span>;
}
