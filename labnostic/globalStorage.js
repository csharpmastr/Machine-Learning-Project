class GlobalStorage {
	static set(key, value) {
	    localStorage.setItem(key, JSON.stringify(value));
	}

	static get(key) {
	const item = localStorage.getItem(key);
	
	return item ? JSON.parse(item) : null
	}

    // constructor() {
    //     this.value = 0;
    //     }
        
    //     set(val) {
    //     this.value = val;
    //     }
        
    //     get(val) {
    //     return this.value = val;
    //     }
}

export {GlobalStorage}