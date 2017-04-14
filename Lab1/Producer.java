package consumer.something;

import java.util.List;
 
/**
 * Producer Class.
 */
class Producer implements Runnable {
 
    private List<Integer> sharedQueue;
    private int maxSize=4; //maximum number of products which sharedQueue can hold at a time.
 
    public Producer(List<Integer> sharedQueue) {
        this.sharedQueue = sharedQueue;
    }
 
    @Override
    public void run() {
        for (int i = 0; i < 20; i++) {
         try {
             produce(i+1);
         } catch (InterruptedException e) {
        	 e.printStackTrace(); 
        	 }
        }
}
 
    private void produce(int i) throws InterruptedException {
    
       synchronized (sharedQueue) {
           //if sharedQuey is full wait until consumer consumes.
           while (sharedQueue.size() == maxSize) {
             System.out.println("Queue is full "
                     + "");
             sharedQueue.wait();
         }
        }
       
       /* 2 Synchronized blocks have been used means before
        * producer produces by entering below synchronized
        * block consumer can consume.  
        */
      
       //as soon as producer produces (by adding in sharedQueue) it notifies consumerThread.
        synchronized (sharedQueue) {  
           System.out.println("Produced : " + i);
           sharedQueue.add(i);
         Thread.sleep((long)(Math.random() * 1000));
         sharedQueue.notify();
        }
    }
}
