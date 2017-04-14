package consumer.something;

import java.util.List;

class Consumer implements Runnable {
    private List<Integer> sharedQueue;
    public Consumer(List<Integer> sharedQueue) {
        this.sharedQueue = sharedQueue;
    }
   
    @Override
    public void run() {
        while (true) {
         try {
             consume();
             Thread.sleep(100);
         } catch (InterruptedException e) { 
        	 e.printStackTrace(); 
        	 }
        }
    }
 
    private void consume() throws InterruptedException {
      
       synchronized (sharedQueue) {
           //if sharedQuey is empty wait until producer produces.
           while (sharedQueue.size() == 0) {
                  System.out.println("Queue is empty "
                               + "");
             sharedQueue.wait();
         }
       }
       
        synchronized (sharedQueue) {
           Thread.sleep((long)(Math.random() * 2000));
         System.out.println("CONSUMED : " + sharedQueue.remove(0));
         sharedQueue.notify();
        }
    }
}
   
