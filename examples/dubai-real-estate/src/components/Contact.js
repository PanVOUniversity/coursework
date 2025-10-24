import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { FaPhone, FaEnvelope, FaMapMarkerAlt, FaClock, FaWhatsapp, FaTelegram } from 'react-icons/fa';
import toast from 'react-hot-toast';
import './Contact.css';

const Contact = () => {
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    message: ''
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);

    // Simulate form submission
    setTimeout(() => {
      toast.success('Спасибо за ваше сообщение! Мы свяжемся с вами в ближайшее время.');
      setFormData({ name: '', email: '', phone: '', message: '' });
      setIsSubmitting(false);
    }, 2000);
  };

  const contactInfo = [
    {
      icon: FaPhone,
      title: 'Телефон',
      details: ['+971 4 123 4567', '+971 50 123 4567'],
      color: '#2d5a27'
    },
    {
      icon: FaEnvelope,
      title: 'Email',
      details: ['info@dubaieliteproperties.ae', 'sales@dubaieliteproperties.ae'],
      color: '#4a7c59'
    },
    {
      icon: FaMapMarkerAlt,
      title: 'Адрес',
      details: ['Sheikh Zayed Road', 'Dubai, UAE'],
      color: '#6b8e6b'
    },
    {
      icon: FaClock,
      title: 'Часы работы',
      details: ['Пн-Пт: 9:00 - 18:00', 'Сб: 10:00 - 16:00'],
      color: '#a8d5a8'
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  return (
    <section id="contact" className="contact">
      <div className="container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Свяжитесь с нами</h2>
          <p className="section-subtitle">
            Готовы начать поиск вашей идеальной недвижимости? Свяжитесь с нами сегодня!
          </p>
        </motion.div>

        <motion.div
          className="contact-content"
          ref={ref}
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          <motion.div className="contact-info" variants={itemVariants}>
            <h3>Контактная информация</h3>
            <p>Мы всегда готовы помочь вам с любыми вопросами по недвижимости в Дубае.</p>
            
            <div className="contact-items">
              {contactInfo.map((item, index) => (
                <motion.div
                  key={index}
                  className="contact-item"
                  whileHover={{ x: 10 }}
                >
                  <div className="contact-icon" style={{ backgroundColor: item.color }}>
                    <item.icon />
                  </div>
                  <div className="contact-details">
                    <h4>{item.title}</h4>
                    {item.details.map((detail, detailIndex) => (
                      <p key={detailIndex}>{detail}</p>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="social-links">
              <h4>Мы в социальных сетях</h4>
              <div className="social-buttons">
                <motion.a
                  href="#"
                  className="social-btn whatsapp"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <FaWhatsapp />
                  WhatsApp
                </motion.a>
                <motion.a
                  href="#"
                  className="social-btn telegram"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <FaTelegram />
                  Telegram
                </motion.a>
              </div>
            </div>
          </motion.div>

          <motion.div className="contact-form-container" variants={itemVariants}>
            <form className="contact-form" onSubmit={handleSubmit}>
              <h3>Отправить сообщение</h3>
              
              <div className="form-group">
                <label htmlFor="name">Имя *</label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  required
                  className="form-input"
                />
              </div>

              <div className="form-group">
                <label htmlFor="email">Email *</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  required
                  className="form-input"
                />
              </div>

              <div className="form-group">
                <label htmlFor="phone">Телефон</label>
                <input
                  type="tel"
                  id="phone"
                  name="phone"
                  value={formData.phone}
                  onChange={handleInputChange}
                  className="form-input"
                />
              </div>

              <div className="form-group">
                <label htmlFor="message">Сообщение *</label>
                <textarea
                  id="message"
                  name="message"
                  value={formData.message}
                  onChange={handleInputChange}
                  required
                  rows="5"
                  className="form-textarea"
                  placeholder="Расскажите нам о ваших потребностях в недвижимости..."
                />
              </div>

              <motion.button
                type="submit"
                className="btn btn-primary form-submit"
                disabled={isSubmitting}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {isSubmitting ? 'Отправляется...' : 'Отправить сообщение'}
              </motion.button>
            </form>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default Contact;
