import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from audio_tts_msgs.action import TTS
import time
from action_msgs.msg import GoalStatus

class TTSClient(Node):

    def __init__(self):
        super().__init__('tts_client')
        self._client = ActionClient(self, TTS, 'say')

    def send_goal(self, text: str, language: str = 'en'):
        self.get_logger().info('Waiting for action server...')
        self._client.wait_for_server()

        goal_msg = TTS.Goal()
        goal_msg.text = text
        goal_msg.language = language

        self.get_logger().info(f'Sending goal: "{text}"')
        self._send_goal_future = self._client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._goal_handle = goal_handle

        # Wait 3 seconds before sending cancel to simulate preemption
        self.get_logger().info('Waiting 3 seconds before canceling goal...')
        time.sleep(7)

        self.get_logger().info('Sending cancel request...')
        cancel_future = goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info('Received feedback')

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Cancel request accepted')
        else:
            self.get_logger().info('Cancel request rejected')


    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status == 2:
            self.get_logger().info(f'Goal was canceled. Result text: "{result.text}"')
        elif status == 4:
            self.get_logger().info(f'Goal was aborted. Result text: "{result.text}"')
        else:
            self.get_logger().info(f'Goal succeeded. Result text: "{result.text}"')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    tts_client = TTSClient()

    # Send a long text so there's enough time to cancel
    long_text = "This is a long text to test cancellation of the text to speech action server. " \
                "We will send a cancel request after a short delay."

    tts_client.send_goal(long_text)

    rclpy.spin(tts_client)

if __name__ == '__main__':
    main()

