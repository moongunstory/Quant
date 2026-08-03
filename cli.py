import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import logging

import pandas as pd

from src.live.exchange.client import get_client
from src.live.exchange.order import place_market_order
from src.collector import symbol_universe, universe_probe, universe_builder, full_collector
from src.backtest import engine, metrics as M, panel as P, validation as V
from src.backtest.evaluate import required_fields, spec_required_fields
from src.backtest.spec import AlphaSpec, load_all
from src.config.backtest_settings import SETTINGS


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run_trade(args):
    client = get_client(args.mode)

    try:
        result = place_market_order(
            client, args.symbol, args.side, args.quantity,
            reduce_only=args.reduce_only,
        )
    except Exception as e:
        print(f"주문 실패: {e}")
        return

    print(result.data())


def run_universe(args):
    symbol_universe.run()


def run_universe_probe(args):
    universe_probe.run()


def run_universe_builder(args):
    universe_builder.run()


def run_full_collector(args):
    full_collector.run(datasets=args.datasets, max_workers=args.workers)


def run_downsample_metrics(args):
    """기존 metrics parquet(5분봉)을 1시간봉으로 일괄 변환(용량 ~1/12).

    패널은 일 단위 집계만 쓰므로 알파/백테스트 값은 사실상 동일하다.
    변환 후 R2 를 쓰면 `python -m src.live.remote_store push` 로 올려야
    Lambda 도 작은 파일을 받는다(키가 같아 자동 교체됨)."""
    from src.collector.full_collector import downsample_metrics_1h

    d = SETTINGS.processed_dir / "metrics"
    files = sorted(d.glob("*.parquet"))
    if not files:
        print(f"변환할 파일 없음: {d}")
        return
    tot_before = tot_after = 0
    for p in files:
        before = p.stat().st_size
        try:
            df = pd.read_parquet(p)
            out = downsample_metrics_1h(df)
            if len(out) < len(df):
                out.to_parquet(p, index=False)
        except Exception as e:
            print(f"  {p.stem:22s} 실패: {e}")
            continue
        after = p.stat().st_size
        tot_before += before
        tot_after += after
        if args.verbose:
            print(f"  {p.stem:22s} {len(df):>8,}행 → {len(out):>8,}행  "
                  f"{before/1e6:8.1f}MB → {after/1e6:8.1f}MB")
    print(f"\n합계 {len(files)}개 파일: {tot_before/1e9:.2f}GB → {tot_after/1e9:.2f}GB "
          f"(절감 {max(tot_before-tot_after,0)/1e9:.2f}GB)")
    print("R2 사용 시 다음으로 업로드: python -m src.live.remote_store push")


def run_universe_refresh(args):
    """월간 유니버스 갱신 체인: 심볼목록 -> 경량 스캔 -> 월별 top-100 스냅샷."""
    from src.collector import universe_maintenance
    universe_maintenance.run()


def _run_one_backtest(spec, rebuild=False):
    """spec -> EngineResult (패널/유니버스 자동 로드)."""
    fields = spec_required_fields(spec.expression, spec.neutralization)
    if SETTINGS.execution == "next_open":
        fields = set(fields) | {"open"}   # 시가 체결 손익 계산용
    panels = P.load_panels_for_bar(fields, bar=spec.bar, rebuild=rebuild)
    close = panels["close"]
    universe = P.build_universe_mask(close.index, close.columns)
    funding_events = P.load_funding_events(rebuild=rebuild)
    return engine.run(spec, panels, universe=universe, funding_events=funding_events)


def _build_panels(fields=None, rebuild=False):
    """fields(생략 시 전체 FIELD_SPECS) 를 빌드/캐시로드하고 결과를 출력."""
    fields = fields or list(P.FIELD_SPECS.keys())
    for f in fields:
        try:
            panel = P.load_panel(f, rebuild=rebuild)
            print(f"  {f:16s} {panel.shape[0]}일 × {panel.shape[1]}코인  "
                  f"({panel.index.min().date()} ~ {panel.index.max().date()})")
        except Exception as e:
            print(f"  {f:16s} 실패: {e}")


def _print_alpha_report(spec, result, m, validate=False, n_perm=500, save_curve=False):
    """알파 하나의 백테스트 결과(메트릭 + 선택적 검증/자본곡선)를 출력."""
    print(f"\n=== 알파: {spec.name} ===")
    print(f"수식: {spec.expression}")
    print(f"중립화: {spec.neutralization} | decay: {spec.decay} | delay: {spec.delay}")
    print("-" * 48)
    print(f"  sharpe        {m['sharpe']:+.3f}")
    print(f"  sharpe_recent {m.get('sharpe_recent', float('nan')):+.3f}  (최근 90일)")
    print(f"  sharpe_hl     {m.get('sharpe_hl', float('nan')):+.3f}  (반감기 90일 가중)")
    print(f"  sortino       {m.get('sortino', float('nan')):+.3f}  (하방위험 대비)")
    print(f"  calmar        {m.get('calmar', float('nan')):+.3f}  (연수익/최대낙폭)")
    print(f"  win_rate      {m.get('win_rate', float('nan')):.3f}  (이익 난 날 비율, 참고용)")
    print(f"  ann_return    {m['ann_return']:+.4f}  (1단위 북 연환산)")
    print(f"  mdd           {m['mdd']:.4f}")
    print(f"  turnover      {m['turnover']:.3f}  (하루 평균)")
    print(f"  ic (avg)      {m['ic']:+.4f}  (1·5·10·20일 평균)")
    print(f"  ic_1d         {m.get('ic_1d', float('nan')):+.4f}")
    print(f"  ic_5d         {m.get('ic_5d', float('nan')):+.4f}")
    print(f"  ic_10d        {m.get('ic_10d', float('nan')):+.4f}")
    print(f"  ic_20d        {m.get('ic_20d', float('nan')):+.4f}")
    print(f"  net_exposure  {m['net_exposure']:.4f}  (~0=중립)")
    print(f"  fitness       {m['fitness']:+.3f}")
    print(f"  거래일수       {m['days']}")

    if validate:
        print("-" * 48)
        try:
            v = V.validate(result.net_pnl, n_perm=n_perm)
            oos, wf, perm = v["oos"], v["walk_forward"], v["permutation"]
            print(f"  OOS   is={oos['is_sharpe']:+.2f} oos={oos['oos_sharpe']:+.2f} "
                  f"비율={oos['oos_is_ratio']:.2f}")
            print(f"  WF    폴드샤프={[round(s,2) for s in wf['fold_sharpes']]} "
                  f"양수비율={wf['pct_positive']:.0%}")
            print(f"  순열   p-value={perm['p_value']:.4f} "
                  f"({'유의' if perm['p_value']<0.05 else '유의하지 않음'})")
        except ValueError as e:
            print(f"  검증 불가(데이터 부족): {e}")

    if save_curve:
        out = SETTINGS.data_dir / f"backtest_curve_{spec.name}.csv"
        pd.DataFrame({"net_pnl": result.net_pnl,
                      "equity": result.equity}).to_csv(out)
        print(f"\n자본곡선 저장: {out}")


def _print_ranking(rows):
    """(name, metrics dict) 목록을 fitness 내림차순 랭킹 표로 출력."""
    rows = sorted(rows, key=lambda r: r[1]["fitness"], reverse=True)
    print(f"\n{'알파':22s} {'sharpe':>8s} {'s_rec90':>8s} {'s_hl':>7s} "
          f"{'sortino':>8s} {'calmar':>7s} {'win%':>6s} "
          f"{'fitness':>8s} {'turnover':>9s} {'mdd':>7s} "
          f"{'ic_1d':>7s} {'ic_5d':>7s} {'ic_10d':>8s} {'ic_20d':>8s}")
    print("-" * 130)
    for name, m in rows:
        print(f"{name:22s} {m['sharpe']:+8.3f} "
              f"{m.get('sharpe_recent', float('nan')):+8.3f} "
              f"{m.get('sharpe_hl', float('nan')):+7.3f} "
              f"{m.get('sortino', float('nan')):+8.3f} "
              f"{m.get('calmar', float('nan')):+7.3f} "
              f"{m.get('win_rate', float('nan')) * 100:6.1f} "
              f"{m['fitness']:+8.3f} "
              f"{m['turnover']:9.3f} {m['mdd']:7.4f} "
              f"{m.get('ic_1d', float('nan')):+7.4f} "
              f"{m.get('ic_5d', float('nan')):+7.4f} "
              f"{m.get('ic_10d', float('nan')):+8.4f} "
              f"{m.get('ic_20d', float('nan')):+8.4f}")


def run_backtest_build_panel(args):
    _build_panels(fields=args.fields, rebuild=args.rebuild)


def run_backtest_test(args):
    spec = AlphaSpec.load(SETTINGS.alphas_dir / f"{args.alpha}.json")
    result = _run_one_backtest(spec, rebuild=args.rebuild)
    m = M.compute(result)
    _print_alpha_report(spec, result, m, validate=args.validate,
                         n_perm=args.n_perm, save_curve=args.save_curve)


def run_backtest_rank(args):
    specs = load_all(SETTINGS.alphas_dir)
    if not specs:
        print("data/strategy/alphas/ 에 알파가 없음")
        return
    if getattr(args, "only", None):
        want = set(args.only)
        found = {s.name for s in specs}
        missing = want - found
        if missing:
            print(f"  경고: data/strategy/alphas/ 에 없는 이름 무시: {sorted(missing)}")
        specs = [s for s in specs if s.name in want]
        if not specs:
            print("  --only 로 지정한 알파가 하나도 없음")
            return
    rows = []
    for spec in specs:
        try:
            m = M.compute(_run_one_backtest(spec, rebuild=args.rebuild))
            rows.append((spec.name, m))
        except Exception as e:
            print(f"  {spec.name}: 실패 {e}")
    _print_ranking(rows)


def run_backtest_corr(args):
    """알파들의 순수익(net-PnL) 상관행렬 + 독립 그룹(클러스터) 출력.

    상관 높은 두 알파는 사실상 같은 알파 → 포트폴리오 분산에 도움 안 됨.
    '독립 그룹 수' = 진짜 서로 다른 베팅이 몇 개인지."""
    from src.backtest import walkforward as WF

    series = WF.collect_alpha_series(rebuild=args.rebuild)
    if args.only:
        keep = set(args.only)
        missing = keep - set(series)
        if missing:
            print(f"  경고: 풀에 없는(또는 백테스트 불가) 이름 무시: {sorted(missing)}")
        series = {n: d for n, d in series.items() if n in keep}
    if len(series) < 2:
        print("  상관 계산에는 알파가 2개 이상 필요")
        return

    pnl = pd.DataFrame({n: d["pnl"] for n, d in series.items()})
    corr = pnl.corr()
    thr = args.threshold
    # 샤프 내림차순(좋은 것부터)으로 대표를 세운다.
    order = sorted(series, key=lambda n: M.sharpe(series[n]["pnl"].dropna()),
                   reverse=True)

    print(f"\n=== |상관| >= {thr} 인 쌍 (겹치는 알파 = 하나만 채택) ===")
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            c = corr.loc[a, b]
            if pd.notna(c) and abs(c) >= thr:
                pairs.append((abs(c), a, b, c))
    if pairs:
        for _, a, b, c in sorted(pairs, reverse=True):
            print(f"  {a:24s} ~ {b:24s}  corr {c:+.2f}")
    else:
        print("  (없음 — 전부 서로 독립적)")

    reps, cluster = [], {}
    for n in order:
        placed = False
        for r in reps:
            c = corr.loc[n, r]
            if pd.notna(c) and abs(c) >= thr:
                cluster[r].append(n)
                placed = True
                break
        if not placed:
            reps.append(n)
            cluster[n] = []
    print(f"\n=== 독립 그룹 {len(reps)}개 (임계 {thr}) — 실질 '독립 베팅' 수 ===")
    for r in reps:
        s = M.sharpe(series[r]["pnl"].dropna())
        dupes = f"   (겹침: {', '.join(cluster[r])})" if cluster[r] else ""
        print(f"  · {r:24s} sharpe {s:+.2f}{dupes}")

    if args.save:
        out = SETTINGS.data_dir / "meta" / "alpha_corr.csv"
        corr.round(4).to_csv(out)
        print(f"\n상관행렬 저장: {out}")


def run_backtest_all(args):
    """일괄 실행: 패널 전체 빌드 -> 모든 알파 백테스트(+검증) -> fitness 랭킹."""
    print("=== 1) 패널 캐시 빌드 ===")
    _build_panels(rebuild=args.rebuild)

    specs = load_all(SETTINGS.alphas_dir)
    if not specs:
        print("\ndata/strategy/alphas/ 에 알파가 없음")
        return

    print(f"\n=== 2) 알파 {len(specs)}개 백테스트 ===")
    rows = []
    for spec in specs:
        try:
            result = _run_one_backtest(spec, rebuild=False)  # 패널은 위에서 이미 빌드/캐시됨
            m = M.compute(result)
            rows.append((spec.name, m))
            _print_alpha_report(spec, result, m, validate=args.validate,
                                 n_perm=args.n_perm, save_curve=args.save_curve)
        except Exception as e:
            print(f"\n  {spec.name}: 실패 {e}")

    print("\n=== 3) fitness 랭킹 ===")
    _print_ranking(rows)


def run_portfolio(args):
    """포트폴리오 config 로드 -> combine -> risk 오버레이 -> stage 리포트 + 최종 메트릭."""
    from src.portfolio.spec import load_portfolio_spec
    from src.portfolio import pipeline as PP
    from src.portfolio import report as RPT

    cfg = load_portfolio_spec(args.config)
    out = PP.build_portfolio(cfg, rebuild=args.rebuild)
    m = out["metrics"]

    print(f"\n=== 포트폴리오: {cfg.name} ===")
    if out.get("directional_on"):
        da = out.get("directional_alphas") or []
        print(f"방향성 규칙: ON  | partial 승격 알파: {da if da else '(자격 통과 알파 없음 — 전부 5:5 중립)'}")
    print(f"알파비중: " + ", ".join(f"{n}={w:.3f}" for n, w in out['alpha_weights'].items()))
    print("-" * 60)
    print(f"  sharpe        {m['sharpe']:+.3f}")
    print(f"  sortino       {m.get('sortino', float('nan')):+.3f}")
    print(f"  calmar        {m.get('calmar', float('nan')):+.3f}")
    print(f"  win_rate      {m.get('win_rate', float('nan')):.3f}  (참고용)")
    print(f"  ann_return    {m['ann_return']:+.4f}")
    print(f"  mdd           {m['mdd']:.4f}")
    print(f"  vol(자산변동률) {m['vol']:.4f}")
    print(f"  turnover      {m['turnover']:.3f}")
    print(f"  net_exposure  {m['net_exposure']:.4f}")
    print(f"  거래일수       {m['days']}")

    print("\n=== 리스크 stage별 성과 ===")
    print(RPT.format_table(out["report"]))
    if args.save_report:
        jp, cp = RPT.save(out["report"], tag=f"portfolio_{cfg.name}")
        print(f"\n리포트 저장: {jp.name}, {cp.name}")


def run_walkforward(args):
    """선택 과정을 시점마다 재실행하는 미래참조-0 검증 -> OOS 요약 + 연도별 Sharpe.

    --config 를 주면 그 portfolio config 의 selection+weighting+risk_pipeline 을
    시점마다 재적용해 '리스크 스택까지' OOS 검증(weight-space, 프로덕션과 동일 회계).
    생략 시 종전 동작(pnl-space 로 선택만 검증)."""
    from src.backtest import walkforward as WF
    from src.portfolio.pipeline import _load_families

    families = _load_families()
    cfg = None
    if getattr(args, "config", None):
        from src.portfolio.spec import load_portfolio_spec
        cfg = load_portfolio_spec(args.config)
        series, pos_panels, master_panels, funding_events = WF.collect_alpha_books(
            rebuild=args.rebuild, families=families)
        wf = WF.run_walkforward_portfolio(
            series, pos_panels, master_panels, funding_events, cfg,
            families=families, rebalance=args.rebalance, window=args.window,
            min_history=args.min_history)
        print(f"\n=== walkforward (out-of-sample) - config '{cfg.name}' "
              f"(selection+weighting+risk 스택까지 재적용) ===")
    else:
        series = WF.collect_alpha_series(rebuild=args.rebuild)
        wf = WF.run_walkforward(
            series, rebalance=args.rebalance, window=args.window,
            min_history=args.min_history, min_fitness=args.min_fitness,
            max_corr=args.max_corr, top_n=args.top_n,
            families=families, max_per_family=args.max_per_family,
            method=args.method,
        )
        print("\n=== walkforward (out-of-sample) ===")
    print(WF.summarize(wf))
    if args.save_report:
        tag = f"walkforward_{cfg.name}" if cfg else "walkforward"
        jp, cp = WF.save_report(wf, series, tag=tag)
        print(f"\n리포트 저장: {jp.name}, {cp.name}")


def run_live(args):
    """라이브(가상/실매매) 사이클: freshness -> target_weights -> orders -> ledger."""
    from src.live import handler as H
    res = H.run_cycle(args.config, mode=args.mode, refresh=args.refresh,
                      rebuild=args.rebuild, max_staleness_days=args.max_staleness_days)
    H._print_summary(res)


def run_bot(args):
    """텔레그램 Polling 봇 구동."""
    from src.live import telegram_bot as TB
    TB.start_polling_bot()


def run_set_webhook(args):
    """텔레그램 웹훅 등록/해제/조회 (Lambda Function URL 연동).

    예)
      python cli.py set-webhook --url https://<id>.lambda-url.<region>.on.aws/
      python cli.py set-webhook --delete
      python cli.py set-webhook --info
    --secret 생략 시 .env 의 TELEGRAM_WEBHOOK_SECRET 를 사용한다(Lambda 도 같은 값이어야 함).
    """
    from src.live import telegram_bot as TB

    if args.info:
        print(json.dumps(TB.get_webhook_info(), indent=2, ensure_ascii=False))
        return
    if args.delete:
        print(json.dumps(TB.delete_webhook(), indent=2, ensure_ascii=False))
        return
    if not args.url:
        print("등록하려면 --url 을 주세요. (해제는 --delete, 상태확인은 --info)")
        return

    secret = args.secret or os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if not secret:
        print("⚠️  TELEGRAM_WEBHOOK_SECRET 이 없습니다. 위장 요청 차단을 위해 .env 에 설정하거나"
              " --secret 로 넘기는 것을 강력히 권장합니다(Lambda 환경변수와 동일 값).")
    res = TB.set_webhook(args.url, secret=secret)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def run_telemetry_send(args):
    """최근 N일 텔레메트리를 zip 으로 묶어 텔레그램으로 전송(크론/람다 월간 스케줄용).

    예) 매월 1일 09:00 KST 자동 전송:
        0 0 1 * *  cd /path/Quant && python cli.py telemetry-send --days 30
    --no-send 를 주면 전송 없이 zip 파일만 생성(경로 출력)."""
    from src.live import ledger as LG
    if args.no_send:
        zip_path = LG.build_telemetry_bundle(days=args.days)
        if zip_path is None:
            print(f"묶을 텔레메트리 파일이 없습니다 (최근 {args.days}일).")
        else:
            print(f"번들 생성 완료(전송 안 함): {zip_path}")
        return
    from src.live import telegram_bot as TB
    ok = TB.send_telemetry_bundle(days=args.days)
    print("텔레메트리 번들 전송:", "성공" if ok else "실패/대상없음")


def run_attribution(args):
    """전송받은 텔레메트리(zip/폴더/json)로 알파·리스크 기여도를 로컬 분석."""
    from src.live import attribution as AT
    AT.run(args.source, out_dir=args.out_dir)



def main():
    _setup_logging()

    parser = argparse.ArgumentParser(description="Quant 프로젝트 진입점")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trade_parser = subparsers.add_parser("trade", help="COIN 매매 실행 (수동 테스트용)")
    trade_parser.add_argument("--mode", choices=["live", "testnet"], default=None,
                              help="실매매/가상매매. 생략하면 .env의 TRADING_MODE를 따름")
    trade_parser.add_argument("--symbol", required=True, help="예: BTCUSDT")
    trade_parser.add_argument("--side", choices=["BUY", "SELL"], required=True)
    trade_parser.add_argument("--quantity", type=float, required=True)
    trade_parser.add_argument("--reduce-only", action="store_true",
                              help="포지션 축소/청산 전용 주문으로 실행")
    trade_parser.set_defaults(func=run_trade)

    universe_parser = subparsers.add_parser("universe", help="심볼 유니버스 갱신 (data/strategy/meta/symbol_list.json 재생성)")
    universe_parser.set_defaults(func=run_universe)

    universe_probe_parser = subparsers.add_parser("universe-probe", help="유니버스 판단용 rolling_score 스캔 (data/market/scan/*.parquet 갱신)")
    universe_probe_parser.set_defaults(func=run_universe_probe)

    universe_builder_parser = subparsers.add_parser("universe-builder", help="과거 전체 리밸런싱 시점의 유니버스 스냅샷 재구성 (data/market/universe/*.json 갱신)")
    universe_builder_parser.set_defaults(func=run_universe_builder)

    universe_refresh_parser = subparsers.add_parser(
        "universe-refresh",
        help="월간 유니버스 갱신 체인: 심볼목록 -> 경량 스캔 -> 월별 top-100 스냅샷 (월 1회 권장)",
    )
    universe_refresh_parser.set_defaults(func=run_universe_refresh)

    full_collector_parser = subparsers.add_parser("full-collector", help="유니버스 편입 이력이 있는 전체 심볼 대상 gap-aware 데이터 수집",)
    full_collector_parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=["klines", "premiumIndexKlines", "metrics", "fundingRate", "bookDepth"],
        help="수집할 데이터셋. 생략 시 기본 4종(klines/premiumIndexKlines/metrics/fundingRate)",
    )
    full_collector_parser.add_argument(
        "--workers", type=int, default=10,
        help="심볼 단위 병렬 워커 수 (기본 10). 요청 간격은 전역 스로틀이 별도로 보장",
    )
    full_collector_parser.set_defaults(func=run_full_collector)

    downsample_parser = subparsers.add_parser(
        "downsample-metrics",
        help="기존 metrics parquet(5분봉)을 1시간봉으로 일괄 변환 — R2 무료한도(10GB) 확보용, 1회 실행",
    )
    downsample_parser.add_argument("--verbose", action="store_true", help="파일별 변환 내역 출력")
    downsample_parser.set_defaults(func=run_downsample_metrics)

    backtest_build_panel_parser = subparsers.add_parser("backtest-build-panel", help="심볼별 parquet -> date×coin 패널 캐시")
    backtest_build_panel_parser.add_argument("--fields", nargs="+", default=None)
    backtest_build_panel_parser.add_argument("--rebuild", action="store_true")
    backtest_build_panel_parser.set_defaults(func=run_backtest_build_panel)

    backtest_test_parser = subparsers.add_parser("backtest-test", help="알파 하나 백테스트")
    backtest_test_parser.add_argument("--alpha", required=True, help="data/strategy/alphas/<이름>.json 의 이름")
    backtest_test_parser.add_argument("--validate", action="store_true", help="OOS/WF/순열검정 실행")
    backtest_test_parser.add_argument("--n-perm", type=int, default=500)
    backtest_test_parser.add_argument("--rebuild", action="store_true", help="패널 캐시 무시하고 재빌드")
    backtest_test_parser.add_argument("--save-curve", action="store_true", help="자본곡선 CSV 저장")
    backtest_test_parser.set_defaults(func=run_backtest_test)

    backtest_rank_parser = subparsers.add_parser("backtest-rank", help="data/strategy/alphas/ 모든 알파 fitness 랭킹")
    backtest_rank_parser.add_argument("--rebuild", action="store_true")
    backtest_rank_parser.add_argument(
        "--only", nargs="+", default=None,
        help="지정한 이름의 알파만 백테스트(새로 추가한 것만 빠르게 확인). "
             "예: --only breakout_uncrowded breakout_lowfunding")
    backtest_rank_parser.set_defaults(func=run_backtest_rank)

    backtest_corr_parser = subparsers.add_parser(
        "backtest-corr", help="알파 간 순수익 상관행렬 + 독립 그룹(클러스터)")
    backtest_corr_parser.add_argument("--rebuild", action="store_true")
    backtest_corr_parser.add_argument("--only", nargs="+", default=None,
                                      help="이 알파들만 비교")
    backtest_corr_parser.add_argument("--threshold", type=float, default=0.5,
                                      help="독립 판정 임계(|corr| 이 값 이상이면 겹침, 기본 0.5)")
    backtest_corr_parser.add_argument("--save", action="store_true",
                                      help="상관행렬 CSV 저장(data/alpha_corr.csv)")
    backtest_corr_parser.set_defaults(func=run_backtest_corr)

    backtest_all_parser = subparsers.add_parser(
        "backtest-all",
        help="일괄 실행: 패널 전체 빌드 -> 모든 알파 백테스트(+검증) -> fitness 랭킹",
    )
    backtest_all_parser.add_argument("--rebuild", action="store_true", help="패널 캐시 무시하고 전체 재빌드")
    backtest_all_parser.add_argument("--validate", action="store_true", help="알파별 OOS/WF/순열검정도 실행")
    backtest_all_parser.add_argument("--n-perm", type=int, default=500)
    backtest_all_parser.add_argument("--save-curve", action="store_true", help="알파별 자본곡선 CSV 저장")
    backtest_all_parser.set_defaults(func=run_backtest_all)

    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="포트폴리오 config(combine+risk) 백테스트 + 리스크 stage 리포트",
    )
    portfolio_parser.add_argument("--config", default="data/strategy/portfolio/config.json",
                                  help="포트폴리오 JSON 경로")
    portfolio_parser.add_argument("--rebuild", action="store_true", help="패널 캐시 재빌드")
    portfolio_parser.add_argument("--save-report", action="store_true",
                                  help="stage 리포트를 logs/ 에 JSON/CSV 저장")
    portfolio_parser.set_defaults(func=run_portfolio)

    walkforward_parser = subparsers.add_parser(
        "walkforward",
        help="선택 과정 미래참조-0 검증: 시점마다 재선택 -> OOS pnl 체인 + 연도별 Sharpe",
    )
    walkforward_parser.add_argument("--config", default=None,
                                    help="portfolio config(JSON) 경로. 주면 그 config 의 "
                                         "selection+weighting+risk_pipeline 을 시점마다 재적용해 "
                                         "리스크 스택까지 OOS 검증(예: data/portfolio_improved.json). "
                                         "생략 시 pnl-space 로 선택만 검증(종전 동작).")
    walkforward_parser.add_argument("--rebalance", type=int, default=63,
                                    help="재선택 주기(일). 기본 63(~분기)")
    walkforward_parser.add_argument("--window", type=int, default=None,
                                    help="점수화 창(일). 생략 시 expanding")
    walkforward_parser.add_argument("--min-history", type=int, default=252,
                                    help="점수화 최소 관측일(기본 252)")
    walkforward_parser.add_argument("--min-fitness", type=float, default=0.0)
    walkforward_parser.add_argument("--max-corr", type=float, default=0.5)
    walkforward_parser.add_argument("--top-n", type=int, default=None)
    walkforward_parser.add_argument("--max-per-family", type=int, default=None,
                                    help="패밀리당 최대 알파 수(data/strategy/meta/alpha_families.json)")
    walkforward_parser.add_argument("--method", default="low_correlation",
                                    help="selection method(low_correlation|manual)")
    walkforward_parser.add_argument("--save-report", action="store_true",
                                    help="OOS JSON + 연도별 Sharpe CSV 저장")
    walkforward_parser.add_argument("--rebuild", action="store_true")
    walkforward_parser.set_defaults(func=run_walkforward)

    live_parser = subparsers.add_parser(
        "live", help="라이브 사이클(가상/실매매): freshness -> 목표가중 -> 주문 -> 원장",
    )
    live_parser.add_argument("--config", default="data/strategy/portfolio/config.json")
    live_parser.add_argument("--mode", choices=["paper", "real"], default=None,
                             help="paper=가상매매(기본), real=실매매/테스트넷. 생략 시 data/runtime/live/config.json 따름")
    live_parser.add_argument("--refresh", action="store_true", help="사이클 전 데이터 최신화(top100×채택필드)")
    live_parser.add_argument("--rebuild", action="store_true", help="패널 캐시 재빌드")
    live_parser.add_argument("--max-staleness-days", type=int, default=None)
    live_parser.set_defaults(func=run_live)

    bot_parser = subparsers.add_parser(
        "bot", help="라이브 텔레그램 Polling 봇 구동"
    )
    bot_parser.set_defaults(func=run_bot)

    set_webhook_parser = subparsers.add_parser(
        "set-webhook", help="텔레그램 웹훅 등록/해제/조회 (Lambda Function URL 연동)"
    )
    set_webhook_parser.add_argument("--url", default=None,
                                    help="Lambda Function URL (예: https://<id>.lambda-url.<region>.on.aws/)")
    set_webhook_parser.add_argument("--secret", default=None,
                                    help="비밀토큰. 생략 시 .env 의 TELEGRAM_WEBHOOK_SECRET 사용")
    set_webhook_parser.add_argument("--delete", action="store_true", help="웹훅 해제(폴링으로 복귀)")
    set_webhook_parser.add_argument("--info", action="store_true", help="현재 웹훅 상태 조회")
    set_webhook_parser.set_defaults(func=run_set_webhook)

    telemetry_send_parser = subparsers.add_parser(
        "telemetry-send",
        help="최근 N일 텔레메트리를 zip 으로 묶어 텔레그램 전송(크론/람다 월간 스케줄용)",
    )
    telemetry_send_parser.add_argument("--days", type=int, default=30,
                                       help="묶을 기간(일). 기본 30")
    telemetry_send_parser.add_argument("--no-send", action="store_true",
                                       help="전송 없이 zip 파일만 생성")
    telemetry_send_parser.set_defaults(func=run_telemetry_send)

    attribution_parser = subparsers.add_parser(
        "attribution",
        help="전송받은 텔레메트리(zip/폴더)로 알파·리스크 기여도 로컬 분석",
    )
    attribution_parser.add_argument("source",
                                    help="telemetry zip / 폴더 / 단일 json 경로")
    attribution_parser.add_argument("--out-dir", default=".",
                                    help="리포트 저장 폴더(기본: 현재 폴더)")
    attribution_parser.set_defaults(func=run_attribution)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
