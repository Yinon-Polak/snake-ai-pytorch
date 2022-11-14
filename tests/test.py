from collections import deque


class TestUpdate:
    def __init__(self):

        self.memory = deque(maxlen=10_000)
        self.max_update_start_steps = 2
        self.max_update_end_steps = 4
        for i in range(50):
            self.memory.append(
                (i, 0, i, 0, 0)
            )

    def update_rewards(self, last_reward: int):
        len_last_trail = 20
        last_records = []
        for _ in range(len_last_trail):
            last_records.insert(0, self.memory.pop())

        modified_last_record = []
        for i, record in enumerate(last_records):
            (state, action, reward, next_state, done) = record
            if i < self.max_update_start_steps or i >= (len_last_trail - self.max_update_end_steps):
                reward = last_reward  # reward
            modified_last_record.append((state, action, reward, next_state, done))
            # self.remember_no_zero(state, action, reward, next_state, done)

        for record in modified_last_record:
            self.memory.append(record)


if __name__ == '__main__':
    tu = TestUpdate()
    tu.update_rewards(-10)
    for i in tu.memory:
        print(i)
