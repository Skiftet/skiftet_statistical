# frozen_string_literal: true

# Integration: drive each policy through a simulated Bernoulli environment and
# confirm it learns to favour the genuinely best arm. Seeded, so deterministic.
RSpec.describe "bandit convergence" do
  def simulate(policy, true_rates:, rounds:, seed:)
    env = Random.new(seed)
    bandit = SkiftetStatistical::Bandit.new(arms: true_rates.keys, policy: policy)
    rounds.times do
      choice = bandit.select
      reward = env.rand < true_rates.fetch(choice) ? 1 : 0
      bandit.record(choice, reward)
    end
    bandit
  end

  let(:true_rates) { { "low" => 0.2, "mid" => 0.5, "best" => 0.8 } }

  it "Thompson Sampling converges to the best arm" do
    policy = SkiftetStatistical::Policies::ThompsonSampling.new(rng: Random.new(1))
    bandit = simulate(policy, true_rates: true_rates, rounds: 2000, seed: 100)
    expect(bandit.best_arm).to eq("best")
    expect(bandit.arm("best").pulls).to be > bandit.arm("low").pulls
  end

  it "UCB1 converges to the best arm" do
    policy = SkiftetStatistical::Policies::UCB1.new(rng: Random.new(1))
    bandit = simulate(policy, true_rates: true_rates, rounds: 2000, seed: 100)
    expect(bandit.best_arm).to eq("best")
    expect(bandit.arm("best").pulls).to be > bandit.arm("low").pulls
  end

  it "epsilon-greedy converges to the best arm" do
    policy = SkiftetStatistical::Policies::EpsilonGreedy.new(epsilon: 0.1, rng: Random.new(1))
    bandit = simulate(policy, true_rates: true_rates, rounds: 2000, seed: 100)
    expect(bandit.best_arm).to eq("best")
  end

  it "concentrates most pulls on the best arm (low regret)" do
    policy = SkiftetStatistical::Policies::ThompsonSampling.new(rng: Random.new(5))
    bandit = simulate(policy, true_rates: true_rates, rounds: 3000, seed: 200)
    total = bandit.arms.sum(&:pulls)
    best_share = bandit.arm("best").pulls.to_f / total
    expect(best_share).to be > 0.7
  end
end
