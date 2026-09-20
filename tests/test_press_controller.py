import unittest

from so101_typing.supervisor.press_controller import (
    PressControllerState,
    PressDirectiveKind,
    PressOutcome,
    SingleKeyPressConfig,
    SingleKeyPressController,
)
from so101_typing.supervisor.verification import (
    ScreenVerificationStatus,
)


class TestSingleKeyPressController(unittest.TestCase):
    def reach_first_verification(self):
        controller = SingleKeyPressController()

        directive = controller.on_alignment(
            aligned=True
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.DESCEND_TO_Z,
        )
        self.assertEqual(
            directive.z_target_mm,
            -3.0,
        )

        directive = (
            controller.on_z_motion_stable()
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.OBSERVE_WRIST,
        )

        directive = controller.on_alignment(
            aligned=True
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.VERIFY_SCREEN,
        )

        return controller

    def test_initial_alignment_authorizes_first_z(self):
        controller = SingleKeyPressController()

        directive = controller.on_alignment(
            aligned=True
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.DESCEND_TO_Z,
        )
        self.assertEqual(
            directive.z_target_mm,
            -3.0,
        )
        self.assertEqual(
            controller.state,
            PressControllerState.WAITING_Z_SETTLE,
        )

    def test_misalignment_requests_xy_without_descent(self):
        controller = SingleKeyPressController()

        directive = controller.on_alignment(
            aligned=False
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.ALIGN_XY,
        )
        self.assertEqual(
            directive.z_target_mm,
            0.0,
        )
        self.assertEqual(
            controller.z_level_index,
            -1,
        )

    def test_no_change_authorizes_second_z(self):
        controller = (
            self.reach_first_verification()
        )

        directive = controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.DESCEND_TO_Z,
        )
        self.assertEqual(
            directive.z_target_mm,
            -6.0,
        )

    def test_uncertain_holds_current_z(self):
        controller = (
            self.reach_first_verification()
        )

        directive = controller.on_verification(
            ScreenVerificationStatus.UNCERTAIN
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.REOBSERVE_SCREEN,
        )
        self.assertEqual(
            directive.z_target_mm,
            -3.0,
        )
        self.assertEqual(
            controller.current_z_mm,
            -3.0,
        )

    def test_success_retracts_then_succeeds(self):
        controller = (
            self.reach_first_verification()
        )

        directive = controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_SUCCESS
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.RETRACT_TO_Z0,
        )
        self.assertEqual(
            controller.pending_outcome,
            PressOutcome.SUCCESS,
        )

        complete = (
            controller.on_retraction_complete()
        )

        self.assertEqual(
            complete.kind,
            PressDirectiveKind.COMPLETE,
        )
        self.assertEqual(
            controller.state,
            PressControllerState.SUCCEEDED,
        )
        self.assertEqual(
            controller.current_z_mm,
            0.0,
        )

    def test_wrong_retracts_then_fails(self):
        controller = (
            self.reach_first_verification()
        )

        controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_WRONG
        )

        controller.on_retraction_complete()

        self.assertEqual(
            controller.state,
            PressControllerState.FAILED,
        )
        self.assertEqual(
            controller.pending_outcome,
            PressOutcome.WRONG_KEY,
        )

    def test_depth_exhaustion_retracts(self):
        controller = (
            self.reach_first_verification()
        )

        controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        )

        controller.on_z_motion_stable()
        controller.on_alignment(
            aligned=True
        )

        directive = controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.RETRACT_TO_Z0,
        )
        self.assertEqual(
            controller.pending_outcome,
            PressOutcome.DEPTH_EXHAUSTED,
        )

    def test_same_z_realign_does_not_authorize_new_depth(self):
        controller = (
            self.reach_first_verification()
        )

        controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        )

        controller.on_z_motion_stable()

        directive = controller.on_alignment(
            aligned=False
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.ALIGN_XY,
        )
        self.assertEqual(
            directive.z_target_mm,
            -6.0,
        )

        directive = controller.on_alignment(
            aligned=True
        )

        self.assertEqual(
            directive.kind,
            PressDirectiveKind.VERIFY_SCREEN,
        )
        self.assertEqual(
            controller.current_z_mm,
            -6.0,
        )

    def test_uncertain_budget_is_bounded(self):
        controller = SingleKeyPressController(
            SingleKeyPressConfig(
                z_levels_mm=(-3.0, -6.0),
                max_uncertain_reobservations=2,
            )
        )

        controller.on_alignment(
            aligned=True
        )
        controller.on_z_motion_stable()
        controller.on_alignment(
            aligned=True
        )

        first = controller.on_verification(
            ScreenVerificationStatus.UNCERTAIN
        )

        second = controller.on_verification(
            ScreenVerificationStatus.UNCERTAIN
        )

        third = controller.on_verification(
            ScreenVerificationStatus.UNCERTAIN
        )

        self.assertEqual(
            first.kind,
            PressDirectiveKind.REOBSERVE_SCREEN,
        )
        self.assertEqual(
            second.kind,
            PressDirectiveKind.REOBSERVE_SCREEN,
        )
        self.assertEqual(
            third.kind,
            PressDirectiveKind.RETRACT_TO_Z0,
        )
        self.assertEqual(
            controller.pending_outcome,
            PressOutcome.UNCERTAIN_EXHAUSTED,
        )

    def test_terminal_controller_rejects_more_descent(self):
        controller = (
            self.reach_first_verification()
        )

        controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_SUCCESS
        )
        controller.on_retraction_complete()

        with self.assertRaises(RuntimeError):
            controller.on_verification(
                ScreenVerificationStatus.CONFIRMED_NO_CHANGE
            )


    def test_side_event_requires_release_before_verification(self):
        controller = SingleKeyPressController(
            SingleKeyPressConfig(
                z_levels_mm=None,
                z_step_mm=2.0,
                max_uncertain_reobservations=1,
            )
        )

        controller.on_alignment(aligned=True)

        release = controller.on_screen_event(
            release_z_target_mm=0.0,
        )

        self.assertEqual(
            release.kind,
            PressDirectiveKind.RELEASE_TO_Z,
        )
        self.assertTrue(controller.screen_event_latched)
        self.assertEqual(
            controller.state,
            PressControllerState.RELEASING,
        )

        verify = controller.on_release_complete()
        self.assertEqual(
            verify.kind,
            PressDirectiveKind.VERIFY_SCREEN,
        )

    def test_latched_side_event_never_descends_after_no_change(self):
        controller = SingleKeyPressController(
            SingleKeyPressConfig(
                z_levels_mm=None,
                z_step_mm=2.0,
                max_uncertain_reobservations=1,
            )
        )

        controller.on_alignment(aligned=True)
        controller.on_screen_event(release_z_target_mm=0.0)
        controller.on_release_complete()

        first = controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        )
        self.assertEqual(
            first.kind,
            PressDirectiveKind.REOBSERVE_SCREEN,
        )

        second = controller.on_verification(
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        )
        self.assertEqual(
            second.kind,
            PressDirectiveKind.RETRACT_TO_Z0,
        )
        self.assertNotEqual(
            second.kind,
            PressDirectiveKind.DESCEND_TO_Z,
        )


if __name__ == "__main__":
    unittest.main()
