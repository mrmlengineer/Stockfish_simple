import argparse
import re
import sys
import pathlib
import os

from testing import (
    EPD,
    TSAN,
    Stockfish as Engine,
    MiniTestFramework,
    OrderedClassMembers,
    Valgrind,
    Syzygy,
)

PATH = pathlib.Path(__file__).parent.resolve()
CWD = os.getcwd()
DEFAULT_NNUEX_FILE = (
    PATH.parent.parent / "export" / "quantized_weights_qat_only3_split_h1_dual_heads_stm16buckets.nnuex"
).resolve()


def get_prefix():
    if args.valgrind:
        return Valgrind.get_valgrind_command()
    if args.valgrind_thread:
        return Valgrind.get_valgrind_thread_command()

    return []


def get_threads():
    if args.valgrind_thread or args.sanitizer_thread:
        return 2
    return 1


def get_path():
    return os.path.abspath(os.path.join(CWD, args.stockfish_path))


def postfix_check(output):
    if args.sanitizer_undefined:
        for idx, line in enumerate(output):
            if "runtime error:" in line:
                # print next possible 50 lines
                for i in range(50):
                    debug_idx = idx + i
                    if debug_idx < len(output):
                        print(output[debug_idx])
                return False

    if args.sanitizer_thread:
        for idx, line in enumerate(output):
            if "WARNING: ThreadSanitizer:" in line:
                # print next possible 50 lines
                for i in range(50):
                    debug_idx = idx + i
                    if debug_idx < len(output):
                        print(output[debug_idx])
                return False

    return True


def Stockfish(*args, **kwargs):
    return Engine(get_prefix(), get_path(), *args, **kwargs)


class TestCLI(metaclass=OrderedClassMembers):
    def beforeAll(self):
        pass

    def afterAll(self):
        pass

    def beforeEach(self):
        self.stockfish = None

    def afterEach(self):
        assert postfix_check(self.stockfish.get_output()) == True
        self.stockfish.clear_output()

    def test_eval(self):
        self.stockfish = Stockfish("eval".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_go_nodes_1000(self):
        self.stockfish = Stockfish("go nodes 1000".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_go_depth_10(self):
        self.stockfish = Stockfish("go depth 10".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_go_perft_4(self):
        self.stockfish = Stockfish("go perft 4".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_go_movetime_1000(self):
        self.stockfish = Stockfish("go movetime 1000".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_go_wtime_8000_btime_8000_winc_500_binc_500(self):
        self.stockfish = Stockfish(
            "go wtime 8000 btime 8000 winc 500 binc 500".split(" "),
            True,
        )
        assert self.stockfish.process.returncode == 0

    def test_go_wtime_1000_btime_1000_winc_0_binc_0(self):
        self.stockfish = Stockfish(
            "go wtime 1000 btime 1000 winc 0 binc 0".split(" "),
            True,
        )
        assert self.stockfish.process.returncode == 0

    def test_go_wtime_1000_btime_1000_winc_0_binc_0_movestogo_5(self):
        self.stockfish = Stockfish(
            "go wtime 1000 btime 1000 winc 0 binc 0 movestogo 5".split(" "),
            True,
        )
        assert self.stockfish.process.returncode == 0

    def test_go_movetime_200(self):
        self.stockfish = Stockfish("go movetime 200".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_go_nodes_20000_searchmoves_e2e4_d2d4(self):
        self.stockfish = Stockfish(
            "go nodes 20000 searchmoves e2e4 d2d4".split(" "), True
        )
        assert self.stockfish.process.returncode == 0

    def test_bench_128_threads_8_default_depth(self):
        self.stockfish = Stockfish(
            f"bench 128 {get_threads()} 8 default depth".split(" "),
            True,
        )
        assert self.stockfish.process.returncode == 0

    def test_bench_128_threads_3_bench_tmp_epd_depth(self):
        self.stockfish = Stockfish(
            f"bench 128 {get_threads()} 3 {os.path.join(PATH, 'bench_tmp.epd')} depth".split(
                " "
            ),
            True,
        )
        assert self.stockfish.process.returncode == 0

    def test_d(self):
        self.stockfish = Stockfish("d".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_compiler(self):
        self.stockfish = Stockfish("compiler".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_license(self):
        self.stockfish = Stockfish("license".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_uci(self):
        self.stockfish = Stockfish("uci".split(" "), True)
        assert self.stockfish.process.returncode == 0

    def test_uci_exposes_nnuex_options(self):
        self.stockfish = Stockfish(
            ["uci"],
            True,
        )

        output = self.stockfish.process.stdout

        assert "option name EvalFile type string default" in output
        assert "option name NNUEXMode type combo default full incremental auto" in output
        assert "option name NNUEXParityCheck type check default false" in output
        assert "option name NNUEXMetrics type check default false" in output


class TestInteractive(metaclass=OrderedClassMembers):
    def beforeAll(self):
        self.stockfish = Stockfish()

    def afterAll(self):
        self.stockfish.quit()
        assert self.stockfish.close() == 0

    def afterEach(self):
        assert postfix_check(self.stockfish.get_output()) == True
        self.stockfish.clear_output()

    def _configure_nnuex_mode(self, mode: str):
        self.stockfish.send_command("setoption name Threads value 1")
        self.stockfish.send_command("setoption name Hash value 16")
        self.stockfish.send_command("setoption name MultiPV value 1")
        self.stockfish.send_command("setoption name UCI_ShowWDL value false")
        self.stockfish.send_command("setoption name NNUEXParityCheck value false")
        self.stockfish.send_command("setoption name NNUEXMetrics value false")
        self.stockfish.send_command(f"setoption name EvalFile value {DEFAULT_NNUEX_FILE}")
        self.stockfish.send_command(f"setoption name NNUEXMode value {mode}")
        self.stockfish.send_command("setoption name Clear Hash")
        self.stockfish.send_command("ucinewgame")

    def _nnuex_eval_signature(self, mode: str, fen: str) -> str:
        self._configure_nnuex_mode(mode)
        self.stockfish.send_command(f"position fen {fen}")
        self.stockfish.send_command("eval")
        self.stockfish.send_command("isready")

        signature = {"final_eval": None}

        def callback(output):
            match = re.match(r"final eval \(white cp\):\s+(.+)", output)
            if match:
                signature["final_eval"] = match.group(1)

            if output == "readyok":
                return True

            return False

        self.stockfish.check_output(callback)
        assert signature["final_eval"] is not None
        return signature["final_eval"]

    def _nnuex_search_signature(self, mode: str, fen: str, depth: int) -> tuple[str, str]:
        self._configure_nnuex_mode(mode)
        self.stockfish.send_command(f"position fen {fen}")
        self.stockfish.send_command(f"go depth {depth}")

        signature = {"score": "", "bestmove": None}

        def callback(output):
            match = re.match(rf"info depth {depth}\b.* score (cp -?\d+|mate -?\d+)\b", output)
            if match:
                signature["score"] = match.group(1)

            if output.startswith("bestmove "):
                parts = output.split()
                signature["bestmove"] = parts[1] if len(parts) >= 2 else ""
                return True

            return False

        self.stockfish.check_output(callback)
        assert signature["bestmove"] is not None
        return signature["score"], signature["bestmove"]

    def test_startup_output(self):
        self.stockfish.starts_with("Stockfish")

    def test_uci_command(self):
        self.stockfish.send_command("uci")
        self.stockfish.equals("uciok")

    def test_set_threads_option(self):
        self.stockfish.send_command(f"setoption name Threads value {get_threads()}")

    def test_ucinewgame_and_startpos_nodes_1000(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command("position startpos")
        self.stockfish.send_command("go nodes 1000")
        self.stockfish.starts_with("bestmove")

    def test_ucinewgame_and_startpos_moves(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command("position startpos moves e2e4 e7e6")
        self.stockfish.send_command("go nodes 1000")
        self.stockfish.starts_with("bestmove")

    def test_fen_position_1(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command("position fen 5rk1/1K4p1/8/8/3B4/8/8/8 b - - 0 1")
        self.stockfish.send_command("go nodes 1000")
        self.stockfish.starts_with("bestmove")

    def test_fen_position_2_flip(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command("position fen 5rk1/1K4p1/8/8/3B4/8/8/8 b - - 0 1")
        self.stockfish.send_command("flip")
        self.stockfish.send_command("go nodes 1000")
        self.stockfish.starts_with("bestmove")

    def test_depth_5_with_callback(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command("position startpos")
        self.stockfish.send_command("go depth 5")

        def callback(output):
            regex = r"info depth \d+ seldepth \d+ multipv \d+ score cp -?\d+ nodes \d+ nps \d+ hashfull \d+ tbhits \d+ time \d+ pv"
            if output.startswith("info depth") and not re.match(regex, output):
                assert False
            if output.startswith("bestmove"):
                return True
            return False

        self.stockfish.check_output(callback)

    def test_ucinewgame_and_go_depth_9(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command("setoption name UCI_ShowWDL value true")
        self.stockfish.send_command("position startpos")
        self.stockfish.send_command("go depth 9")

        depth = 1

        def callback(output):
            nonlocal depth

            regex = rf"info depth {depth} seldepth \d+ multipv \d+ score cp -?\d+ wdl \d+ \d+ \d+ nodes \d+ nps \d+ hashfull \d+ tbhits \d+ time \d+ pv"

            if output.startswith("info depth"):
                if not re.match(regex, output):
                    assert False
                depth += 1

            if output.startswith("bestmove"):
                assert depth == 10
                return True

            return False

        self.stockfish.check_output(callback)

    def test_clear_hash(self):
        self.stockfish.send_command("setoption name Clear Hash")

    def test_fen_position_mate_1(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen 5K2/8/2qk4/2nPp3/3r4/6B1/B7/3R4 w - e6"
        )
        self.stockfish.send_command("go depth 18")

        self.stockfish.expect("* score mate 1 * pv d5e6")
        self.stockfish.equals("bestmove d5e6")

    def test_fen_position_mate_minus_1(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen 2brrb2/8/p7/Q7/1p1kpPp1/1P1pN1K1/3P4/8 b - -"
        )
        self.stockfish.send_command("go depth 18")
        self.stockfish.expect("* score mate -1 *")
        self.stockfish.starts_with("bestmove")

    def test_fen_position_fixed_node(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen 5K2/8/2P1P1Pk/6pP/3p2P1/1P6/3P4/8 w - - 0 1"
        )
        self.stockfish.send_command("go nodes 500000")
        self.stockfish.starts_with("bestmove")

    def test_fen_position_with_mate_go_depth(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen 8/5R2/2K1P3/4k3/8/b1PPpp1B/5p2/8 w - -"
        )
        self.stockfish.send_command("go depth 18 searchmoves c6d7")
        self.stockfish.expect("* score mate 2 * pv c6d7 * f7f5")

        self.stockfish.starts_with("bestmove")

    def test_fen_position_with_mate_go_mate(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen 8/5R2/2K1P3/4k3/8/b1PPpp1B/5p2/8 w - -"
        )
        self.stockfish.send_command("go mate 2 searchmoves c6d7")
        self.stockfish.expect("* score mate 2 * pv c6d7 *")

        self.stockfish.starts_with("bestmove")

    def test_fen_position_with_mate_go_nodes(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen 8/5R2/2K1P3/4k3/8/b1PPpp1B/5p2/8 w - -"
        )
        self.stockfish.send_command("go nodes 500000 searchmoves c6d7")
        self.stockfish.expect("* score mate 2 * pv c6d7 * f7f5")

        self.stockfish.starts_with("bestmove")

    def test_fen_position_depth_27(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen r1b2r1k/pp1p2pp/2p5/2B1q3/8/8/P1PN2PP/R4RK1 w - - 0 18"
        )
        self.stockfish.send_command("go")
        self.stockfish.contains("score mate 1")

        self.stockfish.starts_with("bestmove")

    def test_fen_position_with_mate_go_depth_and_promotion(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen 8/5R2/2K1P3/4k3/8/b1PPpp1B/5p2/8 w - - moves c6d7 f2f1q"
        )
        self.stockfish.send_command("go depth 18")
        self.stockfish.expect("* score mate 1 * pv f7f5")
        self.stockfish.starts_with("bestmove f7f5")

    def test_fen_position_with_mate_go_depth_and_searchmoves(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen 8/5R2/2K1P3/4k3/8/b1PPpp1B/5p2/8 w - -"
        )
        self.stockfish.send_command("go depth 18 searchmoves c6d7")
        self.stockfish.expect("* score mate 2 * pv c6d7 * f7f5")

        self.stockfish.starts_with("bestmove c6d7")

    def test_fen_position_with_moves_with_mate_go_depth_and_searchmoves(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command(
            "position fen 8/5R2/2K1P3/4k3/8/b1PPpp1B/5p2/8 w - - moves c6d7"
        )
        self.stockfish.send_command("go depth 18 searchmoves e3e2")
        self.stockfish.expect("* score mate -1 * pv e3e2 f7f5")
        self.stockfish.starts_with("bestmove e3e2")

    def test_evalfile_loads_nnuex_network(self):
        self.stockfish.send_command(f"setoption name EvalFile value {DEFAULT_NNUEX_FILE}")
        self.stockfish.send_command("position startpos")
        self.stockfish.send_command("go depth 5")
        self.stockfish.starts_with("bestmove")

    def test_nnuex_mode_incremental(self):
        self.stockfish.send_command("setoption name NNUEXMode value incremental")
        self.stockfish.send_command(f"setoption name EvalFile value {DEFAULT_NNUEX_FILE}")
        self.stockfish.send_command("position startpos")
        self.stockfish.send_command("go depth 5")
        self.stockfish.starts_with("bestmove")
        self.stockfish.send_command("setoption name NNUEXMode value full")

    def test_nnuex_mode_auto(self):
        self.stockfish.send_command("setoption name NNUEXMode value auto")
        self.stockfish.send_command(f"setoption name EvalFile value {DEFAULT_NNUEX_FILE}")
        self.stockfish.send_command("position startpos")
        self.stockfish.send_command("go depth 5")
        self.stockfish.starts_with("bestmove")
        self.stockfish.send_command("setoption name NNUEXMode value full")

    def test_nnuex_modes_match_eval_and_search(self):
        fen = "r1bq1rk1/ppp2ppp/2n5/3np3/2B5/2PP1N2/PP3PPP/RNBQ1RK1 w - - 0 8"
        modes = ("full", "incremental", "auto")

        eval_signatures = {
            mode: self._nnuex_eval_signature(mode, fen)
            for mode in modes
        }
        search_signatures = {
            mode: self._nnuex_search_signature(mode, fen, depth=6)
            for mode in modes
        }

        reference_eval = eval_signatures["full"]
        reference_search = search_signatures["full"]

        for mode in modes[1:]:
            assert eval_signatures[mode] == reference_eval
            assert search_signatures[mode] == reference_search

        self.stockfish.send_command("setoption name NNUEXMode value full")

    def test_nnuex_parity_and_metrics(self):
        self.stockfish.send_command(f"setoption name EvalFile value {DEFAULT_NNUEX_FILE}")
        self.stockfish.send_command("setoption name NNUEXMode value incremental")
        self.stockfish.send_command("setoption name NNUEXParityCheck value true")
        self.stockfish.send_command("setoption name NNUEXMetrics value true")
        self.stockfish.send_command("position startpos")
        self.stockfish.send_command("go depth 3")
        self.stockfish.contains("info string NNUEX incremental parity summary")
        self.stockfish.contains("info string NNUEX metrics summary")
        self.stockfish.starts_with("bestmove")
        self.stockfish.send_command("setoption name NNUEXParityCheck value false")
        self.stockfish.send_command("setoption name NNUEXMetrics value false")
        self.stockfish.send_command("setoption name NNUEXMode value full")

    def test_multipv_setting(self):
        self.stockfish.send_command("setoption name MultiPV value 4")
        self.stockfish.send_command("position startpos")
        self.stockfish.send_command("go depth 5")
        self.stockfish.starts_with("bestmove")

    def test_fen_position_with_skill_level(self):
        self.stockfish.send_command("setoption name Skill Level value 10")
        self.stockfish.send_command("position startpos")
        self.stockfish.send_command("go depth 5")
        self.stockfish.starts_with("bestmove")

        self.stockfish.send_command("setoption name Skill Level value 20")


class TestSyzygy(metaclass=OrderedClassMembers):
    def beforeAll(self):
        self.stockfish = Stockfish()

    def afterAll(self):
        self.stockfish.quit()
        assert self.stockfish.close() == 0

    def afterEach(self):
        assert postfix_check(self.stockfish.get_output()) == True
        self.stockfish.clear_output()

    def test_syzygy_setup(self):
        self.stockfish.starts_with("Stockfish")
        self.stockfish.send_command("uci")
        self.stockfish.send_command(
            f"setoption name SyzygyPath value {os.path.join(PATH, 'syzygy')}"
        )
        self.stockfish.expect(
            "info string Found 35 WDL and 35 DTZ tablebase files (up to 4-man)."
        )

    def test_syzygy_bench(self):
        self.stockfish.send_command("bench 128 1 8 default depth")
        self.stockfish.expect("Nodes searched  :*")

    def test_syzygy_position(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command("position fen 4k3/PP6/8/8/8/8/8/4K3 w - - 0 1")
        self.stockfish.send_command("go depth 5")

        def check_output(output):
            if "score cp 20000" in output or "score mate" in output:
                return True

        self.stockfish.check_output(check_output)
        self.stockfish.expect("bestmove *")

    def test_syzygy_position_2(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command("position fen 8/1P6/2B5/8/4K3/8/6k1/8 w - - 0 1")
        self.stockfish.send_command("go depth 5")

        def check_output(output):
            if "score cp 20000" in output or "score mate" in output:
                return True

        self.stockfish.check_output(check_output)
        self.stockfish.expect("bestmove *")

    def test_syzygy_position_3(self):
        self.stockfish.send_command("ucinewgame")
        self.stockfish.send_command("position fen 8/1P6/2B5/8/4K3/8/6k1/8 b - - 0 1")
        self.stockfish.send_command("go depth 5")

        def check_output(output):
            if "score cp -20000" in output or "score mate -" in output:
                return True

        self.stockfish.check_output(check_output)
        self.stockfish.expect("bestmove *")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stockfish with testing options")
    parser.add_argument("--valgrind", action="store_true", help="Run valgrind testing")
    parser.add_argument(
        "--valgrind-thread", action="store_true", help="Run valgrind-thread testing"
    )
    parser.add_argument(
        "--sanitizer-undefined",
        action="store_true",
        help="Run sanitizer-undefined testing",
    )
    parser.add_argument(
        "--sanitizer-thread", action="store_true", help="Run sanitizer-thread testing"
    )

    parser.add_argument(
        "--none", action="store_true", help="Run without any testing options"
    )
    parser.add_argument("stockfish_path", type=str, help="Path to Stockfish binary")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    EPD.create_bench_epd()
    TSAN.set_tsan_option()
    Syzygy.download_syzygy()

    framework = MiniTestFramework()

    # Each test suite will be run inside a temporary directory
    framework.run([TestCLI, TestInteractive, TestSyzygy])

    EPD.delete_bench_epd()
    TSAN.unset_tsan_option()

    if framework.has_failed():
        sys.exit(1)

    sys.exit(0)
