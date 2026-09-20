import unittest

from so101_typing.supervisor.press_controller import (
    PressDirectiveKind,
    PressOutcome,
    SingleKeyPressConfig,
    SingleKeyPressController,
)
from so101_typing.supervisor.verification import (
    ScreenVerificationStatus,
)


class TestIterativePressController(
    unittest.TestCase
):
    def advance_one_no_change(
        self,
        controller,
    ):
        descend = (
            controller.on_alignment(
                aligned=True
            )
            if not controller._has_descended
            else None
        )

        if descend is not None:
            self.assertEqual(
                descend.kind,
                PressDirectiveKind.DESCEND_TO_Z,
            )

        controller.on_z_motion_stable()

        verify = controller.on_alignment(
            aligned=True
        )

        self.assertEqual(
            verify.kind,
            PressDirectiveKind.VERIFY_SCREEN,
        )

        return controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        )

    def test_iterative_mode_keeps_descending_without_total_limit(
        self,
    ):
        controller = (
            SingleKeyPressController(
                SingleKeyPressConfig(
                    z_levels_mm=None,
                    z_step_mm=2.0,
                    max_descent_mm=None,
                )
            )
        )

        directive = (
            controller.on_alignment(
                aligned=True
            )
        )

        self.assertEqual(
            directive.z_target_mm,
            -2.0,
        )

        for expected_next in (
            -4.0,
            -6.0,
            -8.0,
            -10.0,
            -12.0,
            -14.0,
            -16.0,
            -18.0,
            -20.0,
            -22.0,
            -24.0,
        ):
            controller.on_z_motion_stable()

            controller.on_alignment(
                aligned=True
            )

            directive = (
                controller.on_verification(
                    ScreenVerificationStatus.CONFIRMED_NO_CHANGE
                )
            )

            self.assertEqual(
                directive.kind,
                PressDirectiveKind.DESCEND_TO_Z,
            )

            self.assertEqual(
                directive.z_target_mm,
                expected_next,
            )

    def test_optional_iterative_safety_fuse_still_works(
        self,
    ):
        controller = (
            SingleKeyPressController(
                SingleKeyPressConfig(
                    z_levels_mm=None,
                    z_step_mm=2.0,
                    max_descent_mm=6.0,
                )
            )
        )

        directive = (
            controller.on_alignment(
                aligned=True
            )
        )

        self.assertEqual(
            directive.z_target_mm,
            -2.0,
        )

        for expected in (
            -4.0,
            -6.0,
        ):
            controller.on_z_motion_stable()
            controller.on_alignment(
                aligned=True
            )

            directive = (
                controller.on_verification(
                    ScreenVerificationStatus.CONFIRMED_NO_CHANGE
                )
            )

            self.assertEqual(
                directive.z_target_mm,
                expected,
            )

        controller.on_z_motion_stable()
        controller.on_alignment(
            aligned=True
        )

        directive = (
            controller.on_verification(
                ScreenVerificationStatus.CONFIRMED_NO_CHANGE
            )
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.RETRACT_TO_Z0,
        )

        self.assertEqual(
            controller.pending_outcome,
            PressOutcome.DEPTH_EXHAUSTED,
        )

    def test_iterative_success_still_retracts_immediately(
        self,
    ):
        controller = (
            SingleKeyPressController(
                SingleKeyPressConfig(
                    z_levels_mm=None,
                    z_step_mm=2.0,
                    max_descent_mm=None,
                )
            )
        )

        controller.on_alignment(
            aligned=True
        )

        controller.on_z_motion_stable()

        controller.on_alignment(
            aligned=True
        )

        directive = (
            controller.on_verification(
                ScreenVerificationStatus.CONFIRMED_SUCCESS
            )
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.RETRACT_TO_Z0,
        )

        self.assertEqual(
            controller.pending_outcome,
            PressOutcome.SUCCESS,
        )


if __name__ == "__main__":
    unittest.main()
