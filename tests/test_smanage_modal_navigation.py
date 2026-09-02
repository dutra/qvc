import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.app import App
from textual.widgets import Button

from hpc_scripts import smanage


_UNSET = object()


class ModalHarness(App[None]):
    def __init__(self, modal):
        super().__init__()
        self.modal = modal
        self.result = _UNSET

    def on_mount(self) -> None:
        self.push_screen(self.modal, self.store_result)

    def store_result(self, result) -> None:
        self.result = result


def test_cancel_modal_arrows_move_visible_focus_and_enter_confirms():
    async def scenario():
        modal = smanage.ConfirmCancelScreen("24489297", "validation array")
        app = ModalHarness(modal)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            keep = modal.query_one("#keep", Button)
            confirm = modal.query_one("#confirm", Button)
            assert keep.has_focus

            await pilot.press("right")
            await pilot.pause()
            assert confirm.has_focus
            assert confirm.styles.border_top[0] == "heavy"

            await pilot.press("left")
            await pilot.pause()
            assert keep.has_focus

            await pilot.press("down")
            await pilot.pause()
            assert confirm.has_focus

            await pilot.press("enter")
            await pilot.pause()
            assert app.result is True

    asyncio.run(scenario())


def test_log_modal_supports_horizontal_and_vertical_arrow_navigation():
    async def scenario():
        modal = smanage.LogChoiceScreen(Path("stdout.log"), Path("stderr.log"))
        app = ModalHarness(modal)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            stdout = modal.query_one("#stdout", Button)
            stderr = modal.query_one("#stderr", Button)
            cancel = modal.query_one("#cancel", Button)
            assert stdout.has_focus

            await pilot.press("down")
            await pilot.pause()
            assert stderr.has_focus

            await pilot.press("right")
            await pilot.pause()
            assert cancel.has_focus

            await pilot.press("up")
            await pilot.pause()
            assert stderr.has_focus

            await pilot.press("enter")
            await pilot.pause()
            assert app.result == "stderr"

    asyncio.run(scenario())


def test_execute_confirmation_uses_shared_button_navigation():
    execution = {
        "task_id": "24489297_54",
        "task_state": "RUNNING",
        "script": Path("run.sh"),
        "work_dir": Path("."),
        "environment": {"SLURM_JOB_ID": "24489297_54"},
    }

    async def scenario():
        modal = smanage.ConfirmExecuteScreen(execution)
        app = ModalHarness(modal)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert modal.query_one("#keep", Button).has_focus

            await pilot.press("right")
            await pilot.pause()
            assert modal.query_one("#confirm", Button).has_focus

    asyncio.run(scenario())
