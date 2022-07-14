#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <chrono>
#include <algorithm>
#include <atomic>


class ThreadPool {
protected:
	std::mutex lock_;
	std::mutex lock2_;
	std::condition_variable condVar_;
	std::condition_variable condVarDone_;
	bool shutdown_;
	std::queue<std::function<void(void)>> jobs_;
	std::vector<std::thread> threads_;
	std::vector<bool> working_;
	std::atomic<bool> busy_;

public:
	ThreadPool(int threads) : shutdown_(false) {
		// Create the specified number of threads
		threads_.reserve(threads);
		working_.reserve(threads);
		for(int i = 0; i < threads; ++i) {
			threads_.emplace_back(std::bind(&ThreadPool::threadEntry, this, i));
			working_.emplace_back(false);
		}
		busy_ = false;
	}

	~ThreadPool() {
		{
			// Unblock any threads and tell them to stop
			std::unique_lock<std::mutex> l(lock_);

			shutdown_ = true;
			condVar_.notify_all();
		}

		// Wait for all threads to stop
		//std::cerr << "Joining threads" << std::endl;
		for(auto& thread : threads_) {
			thread.join();
		}
	}

	void doJob(std::function<void(void)> func) {
		busy_ = true;
		// Place a job on the queu and unblock a thread
		std::unique_lock<std::mutex> l(lock_);

		jobs_.emplace(std::move(func));
		condVar_.notify_one();
	}

	size_t getSize() {
		return threads_.capacity();
	}

	bool isBusy() {
		lock2_.lock();
		for(bool w : working_) {
			if(w) {
				lock2_.unlock();
				return true;
			}
		}
		lock2_.unlock();
		return false;
	}

	void waitWhileBusy() {
		auto timeout = std::chrono::duration<int, std::milli>(100);
		while(busy_.load()) {
			{
				//std::cerr << "Waiting for threads to finish scheduled work" << std::endl;
				std::unique_lock<std::mutex> l(lock_);
				std::cv_status status = condVarDone_.wait_for(l, timeout);
				if(status == std::cv_status::timeout) {
					std::cerr << "Waiting for threads to finish scheduled work Timedout" << std::endl;
				}
			}
		}
		std::cerr << "Threads have finish all scheduled work" << std::endl;
	}

protected:
	void threadEntry(int i) {
		std::function<void(void)> job;

		while(1) {
			{
				std::unique_lock<std::mutex> l(lock_);

				while(!shutdown_ && jobs_.empty()) {
					working_[i] = false;
					busy_ = isBusy();
					condVarDone_.notify_one();
					condVar_.wait(l);
				}

				if(jobs_.empty()) {
					// No jobs to do and we are shutting down
					std::cerr << "Thread " << i << " terminates" << std::endl;
					return;
				}

				//std::cerr << "Thread " << i << " does a job" << std::endl;
				working_[i] = true;
				job = std::move(jobs_.front());
				jobs_.pop();
			}

			// Do the job without holding any locks
			job();
		}
	}
};